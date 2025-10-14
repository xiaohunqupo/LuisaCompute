#pragma once

#include <cstdlib>
#include <cstring>

#include <vulkan/vulkan.h>

#if defined(LUISA_PLATFORM_WINDOWS)
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#elif defined(LUISA_PLATFORM_APPLE)
#include <vulkan/vulkan_macos.h>
#elif defined(LUISA_PLATFORM_UNIX)
#include <X11/Xlib.h>
#include <vulkan/vulkan_xlib.h>
#if LUISA_ENABLE_WAYLAND
#include <dlfcn.h>
#include <vulkan/vulkan_wayland.h>
#include <wayland-client.h>
#endif
#else
#error "Unsupported platform"
#endif

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/core/magic_enum.h>
#include <luisa/backends/common/volk_init.h>

#define LUISA_CHECK_VULKAN(x)                                            \
    do {                                                                 \
        auto ret = x;                                                    \
        if (ret != VK_SUCCESS) [[unlikely]] {                            \
            if (ret > 0 || ret == VK_ERROR_OUT_OF_DATE_KHR) [[likely]] { \
                LUISA_WARNING_WITH_LOCATION(                             \
                    "Vulkan call `" #x "` returned {} (code = {}).",     \
                    ::luisa::to_string(ret), luisa::to_underlying(ret)); \
            } else [[unlikely]] {                                        \
                LUISA_ERROR_WITH_LOCATION(                               \
                    "Vulkan call `" #x "` failed: {} (code = {}).",      \
                    ::luisa::to_string(ret), luisa::to_underlying(ret)); \
            }                                                            \
        }                                                                \
    } while (false)

namespace luisa::compute {

static constexpr auto LUISA_REQUIRED_VULKAN_VERSION = VK_API_VERSION_1_2;

struct VulkanDeviceUUID {
    uint8_t bytes[VK_UUID_SIZE];
    [[nodiscard]] auto operator==(const VulkanDeviceUUID &rhs) const noexcept {
        return memcmp(bytes, rhs.bytes, sizeof(bytes)) == 0;
    }
};

static_assert(sizeof(VulkanDeviceUUID) == 16u);

class VulkanInstance {

private:
    LCVolkInitializer _volk_initializer;
    VkInstance _instance{nullptr};
    VkDebugUtilsMessengerEXT _debug_messenger{nullptr};

private:
    VulkanInstance() noexcept;

public:
    // disable copy and move
    VulkanInstance(const VulkanInstance &) noexcept = delete;
    VulkanInstance(VulkanInstance &&) noexcept = delete;
    VulkanInstance &operator=(const VulkanInstance &) noexcept = delete;
    VulkanInstance &operator=(VulkanInstance &&) noexcept = delete;
    ~VulkanInstance() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _instance; }
    [[nodiscard]] auto has_debug_layer() const noexcept { return _debug_messenger != nullptr; }

public:
    [[nodiscard]] static luisa::shared_ptr<VulkanInstance> retain() noexcept;
};

}// namespace luisa::compute
