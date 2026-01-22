#pragma once
#include <luisa/vstl/common.h>
#include <luisa/backends/ext/dx_cuda_interop.h>
#include <luisa/backends/ext/vk_cuda_interop.h>

namespace luisa::compute {
class CUDAExternalExt;
class PyInterop {
public:
    Device compute_device;
    vstd::variant<
        DxCudaTimelineEvent,
        TimelineEvent>
        _event;
    uint64_t _event_fence{};
    vstd::variant<
        DxCudaInterop *,
        VkCudaInterop *>
        _ext{};
    uint _render_device_idx;
    CUDAExternalExt* _cu_ext{};
    PyInterop(DeviceInterface *device);
    ~PyInterop();
    void compute_to_render_fence(
        void *signalled_cu_stream_ptr,
        Stream &wait_render_stream);
    void render_to_compute_fence(
        Stream &signalled_render_stream,
        void *wait_cu_stream_ptr);
    BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count);
    ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target);
    void cuda_buffer(uint64_t buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */);
    void unmap(void *cuda_ptr, void *cuda_handle);
private:
    void _init_event();
};
}// namespace luisa::compute