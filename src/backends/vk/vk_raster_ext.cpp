#include "vk_raster_ext.h"
namespace lc::vk {
ResourceCreationInfo VkRasterExt::create_raster_shader(
    Function vert,
    Function pixel,
    const ShaderOption &shader_option) noexcept {
    return ResourceCreationInfo::make_invalid();
}

ResourceCreationInfo VkRasterExt::load_raster_shader(
    luisa::span<Type const *const> types,
    luisa::string_view ser_path) noexcept {
    return ResourceCreationInfo::make_invalid();
}

VkRasterExt::VkRasterExt(Device *device) {
}
VkRasterExt::~VkRasterExt() {}

void VkRasterExt::destroy_raster_shader(uint64_t handle) noexcept {}

// depth buffer
ResourceCreationInfo VkRasterExt::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    return ResourceCreationInfo::make_invalid();
}
void VkRasterExt::destroy_depth_buffer(uint64_t handle) noexcept {}
}// namespace lc::vk