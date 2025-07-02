#include "swapchain.h"
#include <vulkan/vulkan.h>
#include "vk_func_table.h"
#include "log.h"
#include "device.h"

namespace lc::vk {
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    luisa::vector<VkSurfaceFormatKHR> formats;
    luisa::vector<VkPresentModeKHR> present_modes;
};
[[nodiscard]] auto _query_swapchain_support(
    VkFuncTable const &func_table,
    VkPhysicalDevice device, VkSurfaceKHR surface) noexcept {
    SwapChainSupportDetails details;
    func_table.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
    auto format_count = 0u;
    func_table.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
    if (format_count != 0u) {
        details.formats.resize(format_count);
        func_table.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
    }
    auto present_mode_count = 0u;
    func_table.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);
    if (present_mode_count != 0u) {
        details.present_modes.resize(present_mode_count);
        func_table.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
    }
    return details;
}
void _create_surface(
    VkFuncTable const &func_table,
    uint64_t display_handle, uint64_t window_handle, VkSurfaceKHR surface, VkInstance instance) noexcept {
    //TODO: linux wip
    VkWin32SurfaceCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    create_info.hwnd = reinterpret_cast<HWND>(window_handle);
    create_info.hinstance = GetModuleHandle(nullptr);
    VK_CHECK_RESULT(func_table.vkCreateWin32SurfaceKHR(instance, &create_info, Device::alloc_callbacks(), &surface));
}
[[nodiscard]] static auto _is_hdr_colorspace(VkColorSpaceKHR colorspace) noexcept {
    return colorspace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT;
}

[[nodiscard]] static auto _colorspace_name(VkColorSpaceKHR colorspace) noexcept {
    switch (colorspace) {
        case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
            return "sRGB (non-linear)";
        case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT:
            return "Display P3 (non-linear)";
        case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT:
            return "Extended sRGB (linear)";
        case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT:
            return "Display P3 (linear)";
        case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT:
            return "DCI P3 (non-linear)";
        case VK_COLOR_SPACE_BT709_LINEAR_EXT:
            return "BT709 (linear)";
        case VK_COLOR_SPACE_BT709_NONLINEAR_EXT:
            return "BT709 (non-linear)";
        case VK_COLOR_SPACE_BT2020_LINEAR_EXT:
            return "BT2020 (linear)";
        case VK_COLOR_SPACE_HDR10_ST2084_EXT:
            return "HDR10 (ST2084)";
        case VK_COLOR_SPACE_DOLBYVISION_EXT:
            return "Dolby Vision";
        case VK_COLOR_SPACE_HDR10_HLG_EXT:
            return "HDR10 (HLG)";
        case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT:
            return "Adobe RGB (linear)";
        case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT:
            return "Adobe RGB (non-linear)";
        case VK_COLOR_SPACE_PASS_THROUGH_EXT:
            return "Pass-through";
        case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT:
            return "Extended sRGB (non-linear)";
        case VK_COLOR_SPACE_DISPLAY_NATIVE_AMD:
            return "Display Native";
        default:
            break;
    }
    return "Unknown";
}

void Swapchain::create_swapchain(
    uint64_t display_handle,
    uint64_t window_handle,
    uint width, uint height, uint back_buffers,
    bool is_recreation, bool allow_hdr, bool vsync) {
    auto &&func_table = device()->func_table;
    _create_surface(
        func_table,
        display_handle,
        window_handle,
        _surface,
        device()->instance());
    auto support = _query_swapchain_support(
        func_table,
        device()->physical_device(), _surface);
    if (support.capabilities.maxImageCount == 0u) { support.capabilities.maxImageCount = back_buffers; }
    if (!is_recreation) {// only allow change back buffer count and swapchain format on first creation
        back_buffers = std::clamp(
            back_buffers,
            support.capabilities.minImageCount,
            support.capabilities.maxImageCount);
        _swapchain_format = [&formats = support.formats, allow_hdr] {
            for (auto f : formats) {
                LUISA_VERBOSE_WITH_LOCATION(
                    "Supported swapchain format: "
                    "colorspace = {}, format = {}",
                    luisa::to_string(f.colorSpace),
                    luisa::to_string(f.format));
            }
            if (allow_hdr) {
                for (auto format : formats) {
                    if (format.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) { return format; }
                }
            }
            for (auto format : formats) {
                if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
                    (format.format == VK_FORMAT_R8G8B8A8_SRGB ||
                     format.format == VK_FORMAT_B8G8R8A8_SRGB)) { return format; }
            }
            for (auto format : formats) {
                if (format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { return format; }
            }
            return formats.front();
        }();
    }

    _swapchain_extent = [&capabilities = support.capabilities, width, height] {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max() &&
            capabilities.currentExtent.height != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        VkExtent2D actual_extent{width, height};
        actual_extent.width = std::clamp(actual_extent.width,
                                         capabilities.minImageExtent.width,
                                         capabilities.maxImageExtent.width);
        actual_extent.height = std::clamp(actual_extent.height,
                                          capabilities.minImageExtent.height,
                                          capabilities.maxImageExtent.height);
        return actual_extent;
    }();

    auto present_mode = [&present_modes = support.present_modes, vsync] {
        if (!vsync) {// try to use mailbox mode if vsync is disabled
            for (auto mode : present_modes) {
                if (mode == VK_PRESENT_MODE_MAILBOX_KHR) { return mode; }
            }
            for (auto mode : present_modes) {
                if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) { return mode; }
            }
        }
        LUISA_ASSERT(std::find(present_modes.cbegin(),
                               present_modes.cend(),
                               VK_PRESENT_MODE_FIFO_KHR) != present_modes.cend(),
                     "FIFO present mode is not supported.");
        return VK_PRESENT_MODE_FIFO_KHR;
    }();

    // create the swapchain
    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = _surface;
    create_info.minImageCount = back_buffers;
    create_info.imageFormat = _swapchain_format.format;
    create_info.imageColorSpace = _swapchain_format.colorSpace;
    create_info.imageExtent = _swapchain_extent;
    create_info.imageArrayLayers = 1u;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.preTransform = support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    auto logic_device = device()->logic_device();
    VK_CHECK_RESULT(func_table.vkCreateSwapchainKHR(logic_device, &create_info, Device::alloc_callbacks(), &_swapchain));

    // get the swapchain images
    auto image_count = back_buffers;
    _swapchain_images.resize(image_count);
    VK_CHECK_RESULT(func_table.vkGetSwapchainImagesKHR(logic_device, _swapchain, &image_count, _swapchain_images.data()));
    LUISA_ASSERT(image_count == back_buffers, "Swapchain image count mismatch.");

    // create the swapchain image views
    _swapchain_image_views.resize(image_count);
    for (auto i = 0u; i < image_count; i++) {
        VkImageViewCreateInfo image_view_create_info{};
        image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        image_view_create_info.image = _swapchain_images[i];
        image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        image_view_create_info.format = _swapchain_format.format;
        image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_view_create_info.subresourceRange.baseMipLevel = 0u;
        image_view_create_info.subresourceRange.levelCount = 1u;
        image_view_create_info.subresourceRange.baseArrayLayer = 0u;
        image_view_create_info.subresourceRange.layerCount = 1u;
        VK_CHECK_RESULT(vkCreateImageView(logic_device, &image_view_create_info, Device::alloc_callbacks(), &_swapchain_image_views[i]));
    }
    LUISA_INFO("Created swapchain: {}x{} with {} back buffer(s) in {} (format = {}, mode = {}).",
               _swapchain_extent.width, _swapchain_extent.height, back_buffers,
               _colorspace_name(_swapchain_format.colorSpace),
               luisa::to_string(_swapchain_format.format),
               luisa::to_string(present_mode));
}

Swapchain::Swapchain(Device *device)
    : Resource(device) {
}
Swapchain::~Swapchain() {
    for (auto &i : _swapchain_image_views) {
        vkDestroyImageView(
            device()->logic_device(),
            i,
            Device::alloc_callbacks());
    }
    for (auto &i : _swapchain_images) {
        vkDestroyImage(
            device()->logic_device(), i,
            Device::alloc_callbacks());
    }
    device()->func_table.vkDestroySwapchainKHR(device()->logic_device(), _swapchain, Device::alloc_callbacks());
    vkDestroySurfaceKHR(
        device()->instance(),
        _surface,
        Device::alloc_callbacks());
}
}// namespace lc::vk