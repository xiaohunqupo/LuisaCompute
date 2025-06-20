#pragma once
#include "resource.h"
#include <vulkan/vulkan_core.h>
#include "vk_allocator.h"
#include <luisa/runtime/rhi/pixel.h>
namespace lc::vk {
class Texture : public Resource {
    AllocatedImage _img;
    compute::PixelFormat _format;
    uint _mip;
    uint _dimension;
    bool _simultaneous_access;
    mutable luisa::spin_mutex _layout_mtx;
    mutable vstd::vector<VkImageLayout> _layouts;
public:
    auto simultaneous_access() const { return _simultaneous_access; }
    auto dimension() const { return _dimension; }
    Texture(
        Device *device,
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access,
        bool allow_raster_target);
    ~Texture();
    auto mip() const { return _mip; }
    auto vk_image() const { return _img.image; }
    auto format() const { return _format; }
    auto layout(uint level) const {
        std::lock_guard lck{_layout_mtx};
        return _layouts[level];
    }
    auto set_layout(uint level, VkImageLayout layout) const {
        std::lock_guard lck{_layout_mtx};
        _layouts[level] = layout;
    }
    static VkFormat to_vk_format(compute::PixelFormat format);
    Tag tag() const override { return Tag::Texture; }
};
struct TexView {
    Texture const *tex;
    uint level;
    TexView() : tex(nullptr), level(0) {}
    TexView(Texture const *tex) : tex(tex), level(0) {}
    TexView(Texture const *tex, uint level) : tex(tex), level(level) {}
};
}// namespace lc::vk