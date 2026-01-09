//
// Created by mike on 1/10/26.
//

#include "hip_check.h"
#include "hip_texture.h"

namespace luisa::compute::hip {

HIPTexture::HIPTexture() noexcept = default;

HIPTexture::~HIPTexture() noexcept {
    for (auto i = 0u; i < _levels; i++) {
        if (_mip_surfaces[i]) { LUISA_CHECK_HIP(hipDestroySurfaceObject(_mip_surfaces[i])); }
        LUISA_CHECK_HIP(hipFreeArray(_mip_arrays[i]));
    }
    if (_levels > 1u) {
        LUISA_CHECK_HIP(hipFreeMipmappedArray(reinterpret_cast<hipMipmappedArray_t>(_base_array)));
    }
}

hipArray_t HIPTexture::level(uint32_t i) const noexcept {
    LUISA_ASSERT(i < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 i, _levels);
    return _mip_arrays[i];
}

HIPSurface HIPTexture::surface(uint32_t level) const noexcept {
    LUISA_ASSERT(level < _levels,
                 "Invalid level {} for texture with {} level(s).",
                 level, _levels);
    LUISA_ASSERT(!is_block_compressed(format()),
                 "Block compressed textures cannot be used as HIP surfaces.");
    return HIPSurface{_mip_surfaces[level], to_underlying(storage())};
}

namespace {

[[nodiscard]] auto hip_array_format(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8SInt: [[fallthrough]];
        case PixelFormat::RG8SInt: [[fallthrough]];
        case PixelFormat::RGBA8SInt: return HIP_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::R8UInt: [[fallthrough]];
        case PixelFormat::R8UNorm: [[fallthrough]];
        case PixelFormat::RG8UInt: [[fallthrough]];
        case PixelFormat::RG8UNorm: [[fallthrough]];
        case PixelFormat::RGBA8UInt: [[fallthrough]];
        case PixelFormat::RGBA8UNorm: return HIP_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::R16SInt: [[fallthrough]];
        case PixelFormat::RG16SInt: [[fallthrough]];
        case PixelFormat::RGBA16SInt: return HIP_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::R16UInt: [[fallthrough]];
        case PixelFormat::R16UNorm: [[fallthrough]];
        case PixelFormat::RG16UInt: [[fallthrough]];
        case PixelFormat::RG16UNorm: [[fallthrough]];
        case PixelFormat::RGBA16UInt: [[fallthrough]];
        case PixelFormat::RGBA16UNorm: return HIP_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::R32SInt: [[fallthrough]];
        case PixelFormat::RGBA32SInt: return HIP_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::R32UInt: [[fallthrough]];
        case PixelFormat::RG32SInt: [[fallthrough]];
        case PixelFormat::RG32UInt: [[fallthrough]];
        case PixelFormat::RGBA32UInt: return HIP_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::R16F: [[fallthrough]];
        case PixelFormat::RG16F: [[fallthrough]];
        case PixelFormat::RGBA16F: return HIP_AD_FORMAT_HALF;
        case PixelFormat::R32F: [[fallthrough]];
        case PixelFormat::RG32F: [[fallthrough]];
        case PixelFormat::RGBA32F: return HIP_AD_FORMAT_FLOAT;
        case PixelFormat::BC1UNorm: [[fallthrough]];
        case PixelFormat::BC4UNorm: return HIP_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::BC2UNorm: [[fallthrough]];
        case PixelFormat::BC3UNorm: [[fallthrough]];
        case PixelFormat::BC5UNorm: [[fallthrough]];
        case PixelFormat::BC6HUF16: [[fallthrough]];
        case PixelFormat::BC7UNorm: [[fallthrough]];
        case PixelFormat::BC7SRGB: return HIP_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::R10G10B10A2UInt: [[fallthrough]];
        case PixelFormat::R10G10B10A2UNorm: [[fallthrough]];
        case PixelFormat::R11G11B10F: [[fallthrough]];
        case PixelFormat::RGBA8SRGB: return HIP_AD_FORMAT_UNSIGNED_INT8;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid pixel format 0x{:02x}.",
                              luisa::to_underlying(format));
}

[[nodiscard]] auto hip_array_channel_count(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::BC1UNorm: [[fallthrough]];
        case PixelFormat::BC2UNorm: [[fallthrough]];
        case PixelFormat::BC3UNorm: [[fallthrough]];
        case PixelFormat::BC4UNorm: [[fallthrough]];
        case PixelFormat::BC5UNorm: [[fallthrough]];
        case PixelFormat::BC6HUF16: [[fallthrough]];
        case PixelFormat::BC7UNorm: [[fallthrough]];
        case PixelFormat::BC7SRGB: [[fallthrough]];
        case PixelFormat::R10G10B10A2UInt: [[fallthrough]];
        case PixelFormat::R10G10B10A2UNorm: [[fallthrough]];
        case PixelFormat::R11G11B10F: [[fallthrough]];
        case PixelFormat::RGBA8SRGB: return 4u;
        default: break;
    }
    return pixel_format_channel_count(format);
}

}// namespace

HIPTexture *HIPTexture::create_device_texture(PixelFormat format, uint dim, uint3 size, uint32_t mip_levels) noexcept {
    auto t = luisa::new_with_allocator<HIPTexture>();
    t->_size[0] = static_cast<uint16_t>(size.x);
    t->_size[1] = static_cast<uint16_t>(size.y);
    t->_size[2] = static_cast<uint16_t>(size.z);
    t->_format = static_cast<uint8_t>(format);
    t->_levels = static_cast<uint8_t>(mip_levels);
    auto is_bc = is_block_compressed(format);
    HIP_ARRAY3D_DESCRIPTOR array_desc{};
    array_desc.Width = is_bc ? (size.x + 3u) / 4u : size.x;
    array_desc.Height = is_bc ? (size.y + 3u) / 4u : size.y;
    array_desc.Depth = dim == 2u ? 0u : size.z;
    array_desc.Format = hip_array_format(format);
    array_desc.NumChannels = hip_array_channel_count(format);
    if (!is_bc) { array_desc.Flags = hipArraySurfaceLoadStore; }
    if (mip_levels == 1u) {
        hipArray_t array_handle{nullptr};
        LUISA_CHECK_HIP(hipArray3DCreate(&array_handle, &array_desc));
        t->_base_array = array_handle;
        t->_mip_arrays[0] = array_handle;
    } else {
        hipMipmappedArray_t mipmapped_array_handle{nullptr};
        LUISA_CHECK_HIP(hipMipmappedArrayCreate(&mipmapped_array_handle, &array_desc, mip_levels));
        t->_base_array = mipmapped_array_handle;
        for (auto i = 0u; i < mip_levels; i++) {
            hipArray_t level_array{nullptr};
            LUISA_CHECK_HIP(hipGetMipmappedArrayLevel(&level_array, mipmapped_array_handle, i));
            t->_mip_arrays[i] = level_array;
        }
    }
    if (!is_bc) {
        for (auto i = 0u; i < mip_levels; i++) {
            hipResourceDesc res_desc{};
            res_desc.resType = hipResourceTypeArray;
            res_desc.res.array.array = t->_mip_arrays[i];
            LUISA_CHECK_HIP(hipCreateSurfaceObject(&t->_mip_surfaces[i], &res_desc));
        }
    }
    return t;
}

HIPTexture *HIPTexture::import_external_texture(uint64_t external_array, PixelFormat format,
                                                uint dim, uint3 size, uint32_t mip_levels) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
