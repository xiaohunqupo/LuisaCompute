#pragma once
#include <volk.h>
#include "resource.h"
namespace lc::vk {
class Swapchain : public Resource {
    VkSurfaceKHR _surface{};
    VkSurfaceFormatKHR _swapchain_format{};
    VkSampler _texture_sampler{nullptr};
    VkDescriptorSetLayout _descriptor_set_layout{nullptr};
    VkRenderPass _render_pass{};
    VkPipeline _graphics_pipeline{};
    VkExtent2D _swapchain_extent{};
    VkPipelineLayout _pipeline_layout{};
    VkSwapchainKHR _swapchain{};
    luisa::vector<VkImage> _swapchain_images;
    luisa::vector<VkImageView> _swapchain_image_views;
    luisa::vector<VkFramebuffer> _swapchain_framebuffers;
    VkBuffer _vertex_buffer{nullptr};
    VkDeviceMemory _vertex_buffer_memory{nullptr};

public:
    Swapchain(Device *device);
    [[nodiscard]] auto swapchain() const { return _swapchain; }
    ~Swapchain();
    void create_swapchain(
        uint64_t display_handle,
        uint64_t window_handle,
        uint width, uint height, uint back_buffers, bool is_recreation, bool allow_hdr, bool vsync);
    
};
}// namespace lc::vk