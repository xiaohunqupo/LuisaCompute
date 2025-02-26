#include <luisa/core/logging.h>

#include "metal_device.h"
#include "metal_tex_compress.h"

namespace luisa::compute::metal {

#include "metal_tex_compress.inl.h"

static constexpr auto metal_texture_compress_block_size = make_uint3(64u, 1u, 1u);
static constexpr auto metal_texture_compress_thread_count = metal_texture_compress_block_size.x *
                                                           metal_texture_compress_block_size.y *
                                                           metal_texture_compress_block_size.z;

MetalTexCompressExt::MetalTexCompressExt(MetalDevice *device) noexcept {
    auto compile_shader = [device = device->handle()](luisa::string_view f, luisa::string_view s) noexcept {
        LUISA_VERBOSE("Compiling texture compression shader: {}.", f);
        auto source = NS::TransferPtr(NS::String::alloc()->init(
            const_cast<char *>(s.data()), s.size(), NS::UTF8StringEncoding, false));
        auto compile_options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        compile_options->setFastMathEnabled(true);
        compile_options->setOptimizationLevel(MTL::LibraryOptimizationLevelDefault);
        compile_options->setLibraryType(MTL::LibraryTypeExecutable);
        compile_options->setMaxTotalThreadsPerThreadgroup(metal_texture_compress_thread_count);
        compile_options->setLanguageVersion(MTL::LanguageVersion3_0);
        NS::Error *compile_error = nullptr;
        auto library = device->newLibrary(source.get(), compile_options.get(), &compile_error);
        if (compile_error != nullptr) {
            LUISA_WARNING("Errors during texture compression shader compilation: {}.",
                          compile_error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(library, "Failed to compile texture compression shader.");
        auto func_name = NS::TransferPtr(NS::String::alloc()->init(
            const_cast<char *>(f.data()), f.size(), NS::UTF8StringEncoding, false));
        auto func = NS::TransferPtr(library->newFunction(func_name.get()));
        auto pipeline_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
        pipeline_desc->setComputeFunction(func.get());
        pipeline_desc->setMaxTotalThreadsPerThreadgroup(metal_texture_compress_thread_count);
        pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
        pipeline_desc->setLabel(func_name.get());
        NS::Error *pipeline_error = nullptr;
        auto pipeline = NS::TransferPtr(device->newComputePipelineState(
            pipeline_desc.get(), MTL::PipelineOptionNone, nullptr, &pipeline_error));
        if (pipeline_error != nullptr) {
            LUISA_WARNING("Errors during texture compression pipeline creation: {}.",
                          pipeline_error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(pipeline, "Failed to create texture compression pipeline.");
        return pipeline;
    };
    _bc7_encode_try_mode_456 = compile_shader("TryMode456CS", metal_tex_compress_BC7Encode_TryMode456CS);
    _bc7_encode_try_mode_137 = compile_shader("TryMode137CS", metal_tex_compress_BC7Encode_TryMode137CS);
    _bc7_encode_try_mode_02 = compile_shader("TryMode02CS", metal_tex_compress_BC7Encode_TryMode02CS);
    _bc7_encode_encode_block = compile_shader("EncodeBlockCS", metal_tex_compress_BC7Encode_EncodeBlockCS);
    _bc6h_encode_try_mode_g10 = compile_shader("TryModeG10CS", metal_tex_compress_BC6HEncode_TryModeG10CS);
    _bc6h_encode_try_mode_le10 = compile_shader("TryModeLE10CS", metal_tex_compress_BC6HEncode_TryModeLE10CS);
    _bc6h_encode_encode_block = compile_shader("EncodeBlockCS", metal_tex_compress_BC6HEncode_EncodeBlockCS);
}

TexCompressExt::Result MetalTexCompressExt::check_builtin_shader() noexcept {
    return TexCompressExt::Result::Success;
}

TexCompressExt::Result MetalTexCompressExt::compress_bc6h(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result) noexcept {
    return TexCompressExt::compress_bc6h(stream, src, result);
}

TexCompressExt::Result MetalTexCompressExt::compress_bc7(Stream &stream, const ImageView<float> &src, const BufferView<uint> &result, float alpha_importance) noexcept {
    return TexCompressExt::compress_bc7(stream, src, result, alpha_importance);
}

}// namespace luisa::compute::metal
