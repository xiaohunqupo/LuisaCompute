//
// Created by mike on 1/11/26.
//

#pragma once

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include <luisa/core/stl/memory.h>
#include <luisa/runtime/rhi/pixel.h>
#include <luisa/runtime/rhi/resource.h>

namespace luisa::compute {
class VulkanSwapchain;
}// namespace luisa::compute

namespace luisa::compute::hip {

class HIPDevice;
class HIPStream;
class HIPTexture;

class HIPSwapchain {

public:
    class Impl;

public:
    luisa::unique_ptr<Impl> _impl;

public:
    HIPSwapchain(HIPDevice *device, SwapchainOption o) noexcept;
    ~HIPSwapchain() noexcept;
    [[nodiscard]] VulkanSwapchain *native_handle() const noexcept;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept;
    void present(HIPStream *stream, HIPTexture *image) noexcept;
    void set_name(luisa::string name) noexcept;
};

}// namespace luisa::compute::hip

#endif