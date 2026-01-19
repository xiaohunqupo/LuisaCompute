#pragma once
#include <luisa/backends/ext/dx_cuda_interop.h>
#include "stats.h"
namespace lc::validation {
using namespace luisa::compute;
using namespace luisa;
class DxCudaInteropImpl : public DxCudaInterop {
public:
    DeviceStats *device_stats;
    DxCudaInterop *impl;

    BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept override;
    ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept override;
    ResourceCreationInfo create_interop_event() noexcept override;
    void cuda_signal(DeviceInterface *device, uint64_t stream_handle, uint64_t event_handle, uint64_t fence) noexcept override {
        impl->cuda_signal(device, stream_handle, event_handle, fence);
    }
    void cuda_signal(DeviceInterface *device, void *cu_stream_ptr, uint64_t event_handle, uint64_t fence) noexcept override {
        impl->cuda_signal(device, cu_stream_ptr, event_handle, fence);
    }

    void cuda_wait(DeviceInterface *device, uint64_t stream_handle, uint64_t event_handle, uint64_t fence) noexcept override {
        impl->cuda_wait(device, stream_handle, event_handle, fence);
    }

    void cuda_wait(DeviceInterface *device, void *cu_stream_ptr, uint64_t event_handle, uint64_t fence) noexcept override {
        impl->cuda_wait(device, cu_stream_ptr, event_handle, fence);
    }

    void cuda_buffer(uint64_t dx_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept override {
        impl->cuda_buffer(dx_buffer_handle, cuda_ptr, cuda_handle);
    }
    /*CUexternalMemory* */ uint64_t cuda_texture(uint64_t dx_texture_handle) noexcept override {
        return impl->cuda_texture(dx_texture_handle);
    }
    /*CUexternalSemaphore* */ uint64_t cuda_event(uint64_t dx_event_handle) noexcept override {
        return impl->cuda_event(dx_event_handle);
    }
    void destroy_cuda_event(uint64_t cuda_event_handle /*CUexternalSemaphore* */) noexcept override {
        impl->destroy_cuda_event(cuda_event_handle);
    }
    void unmap(void *cuda_ptr, void *cuda_handle) noexcept override {
        impl->unmap(cuda_ptr, cuda_handle);
    }
    DeviceInterface *device() noexcept override {
        return impl->device();
    }
    int cuda_device_index() const noexcept override {
        return impl->cuda_device_index();
    }
};
}// namespace lc::validation