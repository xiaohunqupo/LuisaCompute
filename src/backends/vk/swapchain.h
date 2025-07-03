#pragma once
#include <volk.h>
#include "resource.h"
#include "stream.h"
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
    luisa::vector<VkSemaphore> _image_available_semaphores;
    luisa::vector<VkSemaphore> _render_finished_semaphores;
    luisa::vector<VkFence> _in_flight_fences;

    luisa::vector<VkDescriptorImageInfo> _cached_image_infos;

    size_t _current_frame{0u};

    VkBuffer _vertex_buffer{nullptr};
    VkDeviceMemory _vertex_buffer_memory{nullptr};
    uint64_t _display_handle;
    uint64_t _window_handle;
    uint2 _requested_size;
    bool _requested_hdr{false};
    bool _requested_vsync{false};
    void _recreate_swapchain();
    void _destroy_swapchain();

public:
    Swapchain(Device *device);
    [[nodiscard]] auto swapchain() const { return _swapchain; }
    ~Swapchain();
    void create_swapchain(
        uint64_t display_handle,
        uint64_t window_handle,
        uint width, uint height, uint back_buffers, bool is_recreation, bool allow_hdr, bool vsync);
    void present(
        CommandBuffer &cmdbuffer,
        VkQueue queue,
        VkSemaphore wait, VkSemaphore signal,
        VkImageView image,
        VkImageLayout image_layout,
        VkTimelineSemaphoreSubmitInfo const *timeline_submit_info);
};
}// namespace lc::vk