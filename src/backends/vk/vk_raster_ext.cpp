#include "vk_raster_ext.h"
#include "device.h"
#include "../common/hlsl/shader_compiler.h"
#include "../common/hlsl/hlsl_codegen.h"
#include "shader_serializer.h"
#include "compute_shader.h"
namespace lc::vk {
static constexpr uint k_shader_model = 65u;
ResourceCreationInfo VkRasterExt::create_raster_shader(
    Function vert,
    Function pixel,
    const ShaderOption &option) noexcept {
    LUISA_ASSERT(option.compile_only, "Raster only allow AOT.");
    assert(!option.name.empty());
    uint mask = 0;
    if (option.enable_fast_math) {
        mask |= 1;
    }
    if (option.enable_debug_info) {
        mask |= 2;
    }
    auto code = hlsl::CodegenUtility{}.RasterCodegen(vert, pixel, {}, mask, true);
    vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    auto comp_result = Device::Compiler()->compile_raster(code.result.view(), !option.enable_debug_info, k_shader_model, option.enable_fast_math, true, option.enable_debug_info);
    if (comp_result.vertex.is_type_of<vstd::string>()) [[unlikely]] {
        LUISA_ERROR("DXC compile vertex-shader error: {}", *comp_result.vertex.try_get<vstd::string>());
    }
    if (comp_result.pixel.is_type_of<vstd::string>()) [[unlikely]] {
        LUISA_ERROR("DXC compile pixel-shader error: {}", *comp_result.pixel.try_get<vstd::string>());
    }
    auto kernel_args = [&]() {
        auto vertSpan = vert.arguments();
        auto vertArgs =
            vstd::range_linker{
                vstd::make_ite_range(vertSpan.subspan(1)),
                vstd::transform_range{
                    [&](Variable const &var) {
                        return std::pair<Variable, Usage>{var, vert.variable_usage(var.uid())};
                    }}};
        auto pixelSpan = pixel.arguments();
        auto pixelArgs =
            vstd::range_linker{
                vstd::make_ite_range(pixelSpan.subspan(1)),
                vstd::transform_range{
                    [&](Variable const &var) {
                        return std::pair<Variable, Usage>{var, pixel.variable_usage(var.uid())};
                    }}};
        auto args = vstd::tuple_range(std::move(vertArgs), std::move(pixelArgs)).i_range();
        return ShaderSerializer::serialize_saved_args(args);
    }();
    auto &&vert_buffer = comp_result.vertex.get<0>();
    auto &&pixel_buffer = comp_result.pixel.get<0>();
    ShaderSerializer::serialize_raster(
        code.properties,
        kernel_args,
        check_md5,
        code.typeMD5,
        option.name,
        {reinterpret_cast<const uint *>(vert_buffer->GetBufferPointer()), vert_buffer->GetBufferSize() / sizeof(uint)},
        {reinterpret_cast<const uint *>(pixel_buffer->GetBufferPointer()), pixel_buffer->GetBufferSize() / sizeof(uint)},
        SerdeType::ByteCode,
        _device->binary_io(),
        code.useTex2DBindless,
        code.useTex3DBindless,
        code.useBufferBindless,
        code.printers);
    return ResourceCreationInfo::make_invalid();
}

ResourceCreationInfo VkRasterExt::load_raster_shader(
    luisa::span<Type const *const> types,
    luisa::string_view ser_path) noexcept {
    auto deser_result = ShaderSerializer::try_deser_raster(_device, {}, {}, ser_path, SerdeType::ByteCode, _device->binary_io());
    if (!deser_result.shader)
        return ResourceCreationInfo::make_invalid();
    ResourceCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(deser_result.shader);
    if (!ComputeShader::verify_type_md5(types, deser_result.type_md5)) {
        LUISA_WARNING("Shader {} arguments not match.", name);
        info.invalidate();
        return info;
    }
    return info;
}

VkRasterExt::VkRasterExt(Device *device) {
    _device = device;
}
VkRasterExt::~VkRasterExt() {}

void VkRasterExt::destroy_raster_shader(uint64_t handle) noexcept {
    delete reinterpret_cast<RasterShader*>(handle);
}

// depth buffer
ResourceCreationInfo VkRasterExt::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    return ResourceCreationInfo::make_invalid();
}
void VkRasterExt::destroy_depth_buffer(uint64_t handle) noexcept {}
}// namespace lc::vk