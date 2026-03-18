#include <luisa/runtime/rhi/command.h>
#include <hip/hiprtc.h>
#include "hip_device.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_accel.h"
#include "hip_command_encoder.h"
#include "hip_shader.h"
#include "hip_shader_native.h"
#include "hip_check.h"

namespace luisa::compute::hip {

HIPShaderNative::HIPShaderNative(HIPDevice *device, luisa::string code,
                                 const char *entry, const HIPShaderMetadata &metadata,
                                 luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept
    : HIPShader{metadata.argument_usages},
      _entry{entry},
      _block_size{metadata.block_size.x,
                  metadata.block_size.y,
                  metadata.block_size.z},
      _bound_arguments{std::move(bound_arguments)} {

    hiprtcLinkState link_state{};
    auto create_result = hiprtcLinkCreate(0, nullptr, nullptr, &link_state);
    if (create_result != hiprtcResult::HIPRTC_SUCCESS) {
        LUISA_ERROR_WITH_LOCATION("Failed to create hiprtc link state: {}",
                                  hiprtcGetErrorString(create_result));
    }

    static int dump_count = 0;
    char filename[256];
    snprintf(filename, sizeof(filename), "/tmp/hip_ir_to_test_%d.ll", dump_count);
    FILE *f = fopen(filename, "w");
    if (f) {
        fwrite(code.data(), 1, code.size(), f);
        fclose(f);
    }
    fprintf(stderr, "DEBUG: Dumped IR to %s (%zu bytes)\n", filename, code.size());

    auto add_result = hiprtcLinkAddData(link_state,
                                        hipJitInputLLVMBitcode,
                                        code.data(),
                                        code.size(),
                                        entry,
                                        0, nullptr, nullptr);
    if (add_result != hiprtcResult::HIPRTC_SUCCESS) {
        fprintf(stderr, "DEBUG: hiprtcLinkAddData failed\n");
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

HIPShaderNative::~HIPShaderNative() noexcept {
    LUISA_CHECK_HIP(hipModuleUnload(_module));
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
                LUISA_ERROR_WITH_LOCATION("Accelerator argument not implemented in HIP backend.");
                break;
            }
        }
    };

    fprintf(stderr, "DEBUG: _bound_arguments.size() = %zu\n", _bound_arguments.size());
    fprintf(stderr, "DEBUG: command->arguments().size() = %zu\n", command->arguments().size());
    for (auto &&arg : _bound_arguments) { encode_argument(arg); }
    for (auto &&arg : command->arguments()) { encode_argument(arg); }

    // Encode dispatch size (like CUDA does)
    auto ptr = allocate_argument(sizeof(uint4));
    auto dispatch_size = command->dispatch_size();
    uint4 launch_size_and_kernel_id = make_uint4(dispatch_size, 0u);
    std::memcpy(ptr, &launch_size_and_kernel_id, sizeof(launch_size_and_kernel_id));

    auto hip_stream = encoder.stream()->handle();
    auto block_size = make_uint3(_block_size[0], _block_size[1], _block_size[2]);
    auto blocks = (dispatch_size + block_size - 1u) / block_size;
    fprintf(stderr, "DEBUG: launching kernel with blocks=%u,%u,%u grid, block_size=%u,%u,%u\n",
            blocks.x, blocks.y, blocks.z, block_size.x, block_size.y, block_size.z);
    fprintf(stderr, "DEBUG: argument_buffer offset=%zu, size=%zu\n", argument_buffer_offset, argument_buffer.size());
    void *arguments = argument_buffer.data();
    LUISA_CHECK_HIP(hipModuleLaunchKernel(
        _function,
        blocks.x, blocks.y, blocks.z,
        block_size.x, block_size.y, block_size.z,
        0u, hip_stream, &arguments, nullptr));
}

}// namespace luisa::compute::hip
