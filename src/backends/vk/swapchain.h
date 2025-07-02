#pragma once
#include <volk.h>
#include "resource.h"
namespace lc::vk {
class Swapchain : public Resource {
    VkSurfaceKHR _surface{};
    VkSurfaceFormatKHR _swapchain_format{};
    VkExtent2D _swapchain_extent{};
    VkSwapchainKHR _swapchain{};
    luisa::vector<VkImage> _swapchain_images;
    luisa::vector<VkImageView> _swapchain_image_views;
    public:
    Swapchain(Device *device);

    ~Swapchain();
    void create_swapchain(
        uint64_t display_handle,
        uint64_t window_handle,
        uint width, uint height, uint back_buffers, bool is_recreation, bool allow_hdr, bool vsync);
};
}// namespace lc::vk