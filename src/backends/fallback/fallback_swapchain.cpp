//
// Created by swfly on 2024/12/2.
//

#include <luisa/core/logging.h>

#include "fallback_swapchain.h"
#include "fallback_texture.h"
#include "fallback_stream.h"

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
LUISA_EXPORT_API void *luisa_compute_create_cpu_swapchain(uint64_t display_handle, uint64_t window_handle,
                                                          unsigned width, unsigned height, bool allow_hdr, bool vsync,
                                                          unsigned back_buffer_count) noexcept;
LUISA_EXPORT_API uint8_t luisa_compute_cpu_swapchain_storage(void *swapchain) noexcept;
LUISA_EXPORT_API void *luisa_compute_cpu_swapchain_native_handle(void *swapchain) noexcept;
LUISA_EXPORT_API void luisa_compute_destroy_cpu_swapchain(void *swapchain) noexcept;
LUISA_EXPORT_API void luisa_compute_cpu_swapchain_present(void *swapchain, const void *pixels, uint64_t size) noexcept;
LUISA_EXPORT_API void luisa_compute_cpu_swapchain_present_with_callback(void *swapchain, void *ctx, void (*blit)(void *ctx, void *mapped_pixels)) noexcept;
#endif

namespace luisa::compute::fallback {

FallbackSwapchain::FallbackSwapchain(FallbackStream *bound_stream,
                                     const SwapchainOption &option) noexcept
    : _bound_stream{bound_stream} {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    _handle = luisa_compute_create_cpu_swapchain(option.display, option.window,
                                                 option.size.x, option.size.y,
                                                 option.wants_hdr, option.wants_vsync,
                                                 option.back_buffer_count);
    size = option.size;
#else
    LUISA_ERROR_WITH_LOCATION("Vulkan swapchain is not enabled in fallback backend.");
#endif
}

FallbackSwapchain::~FallbackSwapchain() noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    luisa_compute_destroy_cpu_swapchain(_handle);
#else
    LUISA_ERROR_WITH_LOCATION("Vulkan swapchain is not enabled in fallback backend.");
#endif
}

void FallbackSwapchain::present(FallbackStream *stream, FallbackTexture *frame) {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    LUISA_ASSERT(stream == _bound_stream, "Stream mismatch.");
    stream->queue()->enqueue([handle = this->_handle, frame] {
        auto view = frame->view(0);
        luisa_compute_cpu_swapchain_present(handle, view.data(), view.size_bytes());
    });
    stream->synchronize();
#else
    LUISA_ERROR_WITH_LOCATION("Vulkan swapchain is not enabled in fallback backend.");
#endif
}

}// namespace luisa::compute::fallback
