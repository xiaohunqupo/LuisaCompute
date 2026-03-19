#include <luisa/runtime/rhi/command.h>
#include <hip/hiprtc.h>
#include <hiprt/hiprt.h>
#include "hip_device.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_accel.h"
#include "hip_command_encoder.h"
#include "hip_shader.h"
#include "hip_shader_native.h"
#include "hip_check.h"
#include "hip_rt_wrapper_bitcode_embedded.h"

#ifdef LUISA_COMPUTE_ENABLE_LLVM
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#endif

namespace luisa::compute::hip {

HIPShaderNative::HIPShaderNative(HIPDevice *device, luisa::string code,
                                 const char *entry, const HIPShaderMetadata &metadata,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : HIPShader{metadata.argument_usages},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _is_rt{false} {

    hiprtcLinkState link_state{};
    auto create_result = hiprtcLinkCreate(0, nullptr, nullptr, &link_state);
    if (create_result != hiprtcResult::HIPRTC_SUCCESS) {
        LUISA_ERROR_WITH_LOCATION("Failed to create hiprtc link state: {}",
                                  hiprtcGetErrorString(create_result));
    }

    auto add_result = hiprtcLinkAddData(link_state,
                                        hipJitInputLLVMBitcode,
                                        code.data(),
                                        code.size(),
                                        entry,
                                        0, nullptr, nullptr);
    if (add_result != hiprtcResult::HIPRTC_SUCCESS) {
        hiprtcLinkDestroy(link_state);
        LUISA_ERROR_WITH_LOCATION("Failed to add LLVM bitcode to hiprtc linker: {}",
                                  hiprtcGetErrorString(add_result));
    }

    void *linked_binary = nullptr;
    size_t linked_binary_size = 0;
    auto complete_result = hiprtcLinkComplete(link_state, &linked_binary, &linked_binary_size);
    if (complete_result != hiprtcResult::HIPRTC_SUCCESS) {
        hiprtcLinkDestroy(link_state);
        LUISA_ERROR_WITH_LOCATION("Failed to complete hiprtc linking: {}",
                                  hiprtcGetErrorString(complete_result));
    }

    auto ret = hipModuleLoadData(&_module, linked_binary);
    hiprtcLinkDestroy(link_state);
    if (ret != hipSuccess) {
        LUISA_ERROR_WITH_LOCATION("Failed to load HIP module: {}", hipGetErrorString(ret));
    }
    LUISA_CHECK_HIP(hipModuleGetFunction(&_function, _module, entry));
}

HIPShaderNative::HIPShaderNative(HIPDevice *device, luisa::string code,
                                 const char *entry, const HIPShaderMetadata &metadata,
                                 hiprtContext hiprt_ctx,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : HIPShader{metadata.argument_usages},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)},
      _is_rt{true} {

    llvm::LLVMContext llvm_ctx;
    llvm::SMDiagnostic diag;
    auto kernel_module = llvm::parseAssemblyString(
        llvm::StringRef{code.data(), code.size()}, diag, llvm_ctx);
    if (!kernel_module) {
        LUISA_ERROR_WITH_LOCATION("Failed to parse LLVM IR for RT kernel: {}",
                                  diag.getMessage().str());
    }

    llvm::StringRef wrapper_bc{
        reinterpret_cast<const char *>(luisa_compute_hip_hip_rt_wrapper),
        luisa_compute_hip_hip_rt_wrapper_size};
    auto wrapper_buf = llvm::MemoryBuffer::getMemBuffer(wrapper_bc, "hip_rt_wrapper", false);
    auto wrapper_module = llvm::parseBitcodeFile(*wrapper_buf, llvm_ctx);
    if (!wrapper_module) {
        LUISA_ERROR_WITH_LOCATION("Failed to parse HIPRT wrapper bitcode.");
    }

    if (auto *wrapper_flags = (*wrapper_module)->getNamedMetadata("llvm.module.flags")) {
        wrapper_flags->eraseFromParent();
    }

    if (llvm::Linker::linkModules(*kernel_module, std::move(*wrapper_module))) {
        LUISA_ERROR_WITH_LOCATION("Failed to link kernel module with HIPRT wrapper bitcode.");
    }

    for (auto &func : *kernel_module) {
        auto attrs = func.getAttributes().removeAttributeAtIndex(
            func.getContext(), llvm::AttributeList::FunctionIndex,
            llvm::Attribute::NoCreateUndefOrPoison);
        func.setAttributes(attrs);
        for (auto &bb : func) {
            for (auto &inst : bb) {
                if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&inst)) {
                    auto cb_attrs = cb->getAttributes().removeAttributeAtIndex(
                        cb->getContext(), llvm::AttributeList::FunctionIndex,
                        llvm::Attribute::NoCreateUndefOrPoison);
                    cb->setAttributes(cb_attrs);
                }
            }
        }
    }

    std::string bitcode_str;
    llvm::raw_string_ostream bitcode_os{bitcode_str};
    llvm::WriteBitcodeToFile(*kernel_module, bitcode_os);
    bitcode_os.flush();

    LUISA_INFO("Combined kernel+wrapper linked to {} bytes of bitcode.", bitcode_str.size());

    const char *func_name = entry;
    hiprtApiFunction api_func = nullptr;
    auto hiprt_err = hiprtBuildTraceKernelsFromBitcode(
        hiprt_ctx,
        1u,
        &func_name,
        "rt_module",
        bitcode_str.data(),
        bitcode_str.size(),
        0u,
        1u,
        nullptr,
        &api_func,
        false);
    if (hiprt_err != hiprtSuccess) {
        LUISA_ERROR_WITH_LOCATION(
            "hiprtBuildTraceKernelsFromBitcode failed with error code {}.",
            static_cast<int>(hiprt_err));
    }

    _function = reinterpret_cast<hipFunction_t>(api_func);
    _module = nullptr;

    LUISA_INFO("RT shader compiled successfully via HIPRT.");
}

HIPShaderNative::~HIPShaderNative() noexcept {
    if (_module != nullptr) {
        LUISA_CHECK_HIP(hipModuleUnload(_module));
    }
}

void HIPShaderNative::_launch(HIPCommandEncoder &encoder, ShaderDispatchCommand *command) const noexcept {

    static thread_local std::array<std::byte, 65536u> argument_buffer;

    auto argument_buffer_offset = static_cast<size_t>(0u);
    auto allocate_argument = [&](size_t bytes) noexcept {
        static constexpr auto alignment = 16u;
        auto offset = (argument_buffer_offset + alignment - 1u) / alignment * alignment;
        LUISA_ASSERT(offset + bytes <= argument_buffer.size(),
                     "Too many arguments in ShaderDispatchCommand");
        argument_buffer_offset = offset + bytes;
        return argument_buffer.data() + offset;
    };

    auto encode_argument = [&allocate_argument, command](const auto &arg) noexcept {
        using Tag = ShaderDispatchCommand::Argument::Tag;
        switch (arg.tag) {
            case Tag::BUFFER: {
                auto buffer = reinterpret_cast<const HIPBuffer *>(arg.buffer.handle);
                auto binding = buffer->binding(arg.buffer.offset, arg.buffer.size);
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::TEXTURE: {
                auto texture = reinterpret_cast<HIPTexture *>(arg.texture.handle);
                auto binding = texture->binding(arg.texture.level);
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::UNIFORM: {
                auto uniform = command->uniform(arg.uniform);
                LUISA_ASSERT(arg.uniform.alignment <= 16u, "Invalid uniform alignment {}.",
                             arg.uniform.alignment);
                auto ptr = allocate_argument(uniform.size_bytes());
                std::memcpy(ptr, uniform.data(), uniform.size_bytes());
                break;
            }
            case Tag::BINDLESS_ARRAY: {
                auto array = reinterpret_cast<HIPBindlessArray *>(arg.bindless_array.handle);
                auto binding = array->binding();
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
            case Tag::ACCEL: {
                auto accel = reinterpret_cast<const HIPAccel *>(arg.accel.handle);
                auto binding = accel->binding();
                auto ptr = allocate_argument(sizeof(binding));
                std::memcpy(ptr, &binding, sizeof(binding));
                break;
            }
        }
    };

    for (auto &&arg : _bound_arguments) { encode_argument(arg); }
    for (auto &&arg : command->arguments()) { encode_argument(arg); }

    auto ptr = allocate_argument(sizeof(uint4));
    auto dispatch_size = command->dispatch_size();
    uint4 launch_size_and_kernel_id = make_uint4(dispatch_size, 0u);
    std::memcpy(ptr, &launch_size_and_kernel_id, sizeof(launch_size_and_kernel_id));

    auto hip_stream = encoder.stream()->handle();
    auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
    auto blocks = (dispatch_size + block_size - 1u) / block_size;
    void *arguments = argument_buffer.data();
    LUISA_CHECK_HIP(hipModuleLaunchKernel(
        _function,
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z,
        0u, hip_stream, &arguments, nullptr));
}

}// namespace luisa::compute::hip
