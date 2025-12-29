#pragma once
#include <luisa/backends/ext/vk_cuda_interop.h>
namespace lc::validation {
using namespace luisa::compute;
using namespace luisa;
class VkCudaInteropImpl : public VkCudaInterop {
public:
    VkCudaInterop *impl;

    BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept;
    ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept;
    void _vk_signal(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override {
        impl->_vk_signal(cuda_event_handle, vk_stream, fence_index);
    }
    void _vk_wait(uint64_t cuda_event_handle, uint64_t vk_stream, uint64_t fence_index) noexcept override {
        impl->_vk_wait(cuda_event_handle, vk_stream, fence_index);
    }
    CudaDeviceConfigExt::ExternalVkDevice get_external_vk_device() const noexcept override {
        return impl->get_external_vk_device();
    }
    void cuda_buffer(uint64_t vk_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept override {
        impl->cuda_buffer(vk_buffer_handle, cuda_ptr, cuda_handle);
    }
    uint64_t cuda_texture(uint64_t vk_texture_handle) noexcept override {
        return impl->cuda_texture(vk_texture_handle);
    }
    void unmap(void *cuda_ptr, void *cuda_handle) noexcept override {
        impl->unmap(cuda_ptr, cuda_handle);
    }
    int cuda_device_index() const noexcept override {
        return impl->cuda_device_index();
    }
    DeviceInterface *device() noexcept override {
        return impl->device();
    }
};
}// namespace lc::validation