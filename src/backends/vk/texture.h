#pragma once
#include "resource.h"
#include <volk.h>
#include "vk_allocator.h"
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/depth_format.h>
namespace lc::vk {
class Texture : public Resource {
    AllocatedImage _img;
    compute::PixelFormat _format;
    uint3 _size;
    uint _mip;
    uint _dimension;
    bool _contained : 1 {true};
    bool _simultaneous_access : 1;
    mutable luisa::spin_mutex _layout_mtx;
    mutable vstd::vector<VkImageLayout> _layouts;
public:
    auto simultaneous_access() const { return _simultaneous_access; }
    auto dimension() const { return _dimension; }
    Texture(Device *device);
    // external
    Texture(
        Device *device,
        VkImage external_image,
        uint dimension,
        VkFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access);
    Texture(
        Device *device,
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access,
        bool allow_raster_target);
    ~Texture();
    void init_as_sparse(
        uint dimension,
        compute::PixelFormat format,
        uint3 size,
        uint mip,
        bool simultaneous_access);
    static uint2 tex2d_tile_size(luisa::compute::PixelStorage storage);
    static uint3 tex3d_tile_size(luisa::compute::PixelStorage storage);
    uint3 tile_size() const {
        if (luisa::to_underlying(_format) > 65535u) return {};// depth
        if (_dimension <= 2) {
            return make_uint3(tex2d_tile_size(luisa::compute::pixel_format_to_storage(_format)), 1);
        } else {
            return tex3d_tile_size(luisa::compute::pixel_format_to_storage(_format));
        }
    }
    auto size() const { return _size; }

    auto mip() const { return _mip; }
    auto vk_image() const { return _img.image; }
    auto format() const {
        if (luisa::to_underlying(_format) > 65535u) return static_cast<compute::PixelFormat>(-1);// depth
        return _format;
    }
    auto depth_format() const {
        if (luisa::to_underlying(_format) <= 65535u) return compute::DepthFormat::None;
        return static_cast<compute::DepthFormat>(luisa::to_underlying(_format) & 65535u);
    }
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