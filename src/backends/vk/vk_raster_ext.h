#pragma once
#include <luisa/backends/ext/raster_ext_interface.h>
namespace lc::vk {
class Device;
using namespace luisa;
using namespace luisa::compute;
class VkRasterExt : public RasterExt {
    Device *_device;
public:
    VkRasterExt(Device *device);
    ~VkRasterExt();
    ResourceCreationInfo create_raster_shader(
        Function vert,
        Function pixel,
        const ShaderOption &shader_option) noexcept override;

    ResourceCreationInfo load_raster_shader(
        luisa::span<Type const *const> types,
        luisa::string_view ser_path) noexcept override;

    void destroy_raster_shader(uint64_t handle) noexcept override;

    // depth buffer
    ResourceCreationInfo create_depth_buffer(DepthFormat format, uint width, uint height) noexcept override;
    void destroy_depth_buffer(uint64_t handle) noexcept override;
};
}// namespace lc::vk