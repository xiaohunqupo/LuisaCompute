#pragma once
#include "dx_cuda_timeline_event.h"
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/volume.h>
#include <luisa/backends/ext/native_resource_ext.hpp>
#ifndef NDEBUG
#include <luisa/core/logging.h>
#endif
namespace luisa::compute {


class DxCudaInterop : public DeviceExtension {
    friend class DxCudaTimelineEvent;
public:
    static constexpr luisa::string_view name = "DxCudaInterop";

public:// Don't protect it. The oidn ext is using these interfaces.
    [[nodiscard]] virtual BufferCreationInfo create_interop_buffer(const Type *element, size_t elem_count) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo create_interop_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept = 0;
    [[nodiscard]] virtual ResourceCreationInfo create_interop_event() noexcept = 0;
    virtual void cuda_signal(uint64_t stream_handle, void *event_handle, uint64_t fence) noexcept = 0;
    virtual void cuda_signal(/*CUStream*/ void *cu_stream_ptr, void *event_handle, uint64_t fence) noexcept = 0;
    virtual void cuda_wait(uint64_t stream_handle, void *event_handle, uint64_t fence) noexcept = 0;
    virtual void cuda_wait(/*CUStream*/ void *cu_stream_ptr, void *event_handle, uint64_t fence) noexcept = 0;

public:
    virtual void cuda_buffer(uint64_t dx_buffer_handle, uint64_t *cuda_ptr, uint64_t *cuda_handle /*CUexternalMemory* */) noexcept = 0;
    [[nodiscard]] virtual /*CUexternalMemory* */ uint64_t cuda_texture(uint64_t dx_texture_handle) noexcept = 0;
    [[nodiscard]] virtual /*CUexternalSemaphore* */ void *cuda_event(uint64_t dx_event_handle) noexcept = 0;
    virtual void destroy_cuda_event(void *cuda_event_handle /*CUexternalSemaphore* */) noexcept = 0;
    virtual void unmap(void *cuda_ptr, void *cuda_handle) noexcept = 0;
    virtual DeviceInterface *device() noexcept = 0;
    [[nodiscard]] virtual int cuda_device_index() const noexcept = 0;
    template<typename T>
    Buffer<T> create_buffer(size_t elem_count) noexcept {
        return Buffer<T>{device(), create_interop_buffer(Type::of<T>(), elem_count)};
    }
    ByteBuffer create_byte_buffer(size_t size_bytes) noexcept {
        return ByteBuffer{device(), create_interop_buffer(Type::of<void>(), size_bytes)};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint width, uint height, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, width, height, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint2(width, height),
            mip_levels};
    }
    template<typename T>
    Image<T> create_image(PixelStorage pixel, uint2 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Image<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 2, size.x, size.y, 1, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }
    template<typename T>
    Volume<T> create_volume(PixelStorage pixel, uint width, uint height, uint volume, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, width, height, volume, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            uint3(width, height, volume),
            mip_levels};
    }
    template<typename T>
    Volume<T> create_volume(PixelStorage pixel, uint3 size, uint mip_levels = 1u, bool simultaneous_access = false, bool allow_raster_target = false) noexcept {
        return Volume<T>{
            device(),
            create_interop_texture(pixel_storage_to_format<T>(pixel), 3, size.x, size.y, size.z, mip_levels, simultaneous_access, allow_raster_target),
            pixel,
            size,
            mip_levels};
    }

    virtual ~DxCudaInterop() noexcept = default;
    [[nodiscard]] inline auto create_timeline_event() noexcept {
        return DxCudaTimelineEvent{this};
    }

private:
    [[nodiscard]] inline auto _create_timeline_event() noexcept {
        return TimelineEvent{
            device(),
            create_interop_event()};
    }
};
inline void dx_cuda_interop::Signal::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->cuda_signal(stream_handle, native_handle, fence);
}
inline void dx_cuda_interop::Wait::operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept {
    ext->cuda_wait(stream_handle, native_handle, fence);
}
LUISA_MARK_STREAM_EVENT_TYPE(dx_cuda_interop::Signal)
LUISA_MARK_STREAM_EVENT_TYPE(dx_cuda_interop::Wait)

inline DxCudaTimelineEvent::DxCudaTimelineEvent(DxCudaInterop *ext) noexcept
    : _ext{ext},
      dx_event{ext->_create_timeline_event()},
      _cuda_event{ext->cuda_event(dx_event.handle())} {
}
inline DxCudaTimelineEvent::~DxCudaTimelineEvent() noexcept {
    if (*this) {
        _ext->destroy_cuda_event(_cuda_event);
    }
}
inline void DxCudaTimelineEvent::cuda_signal_external(DxCudaInterop *interop_ext, void *cu_stream_ptr, uint64_t fence) const noexcept {
    interop_ext->cuda_signal(cu_stream_ptr, _cuda_event, fence);
}
inline void DxCudaTimelineEvent::cuda_wait_external(DxCudaInterop *interop_ext, void *cu_stream_ptr, uint64_t fence) const noexcept {
    interop_ext->cuda_wait(cu_stream_ptr, _cuda_event, fence);
}
}// namespace luisa::compute
