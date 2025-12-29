#include "dx_cuda_interop_impl.h"
#include "buffer.h"
#include "texture.h"
#include "event.h"
namespace lc::validation {
BufferCreationInfo DxCudaInteropImpl::create_interop_buffer(const Type *element, size_t elem_count) noexcept {
    auto buffer = impl->create_interop_buffer(element, elem_count);
    new Buffer{buffer.handle, 0};
    return buffer;
}
ResourceCreationInfo DxCudaInteropImpl::create_interop_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept {
    auto tex = impl->create_interop_texture(format, dimension, width, height, depth, mipmap_levels, simultaneous_access, allow_raster_target);
    new Texture{tex.handle, dimension, simultaneous_access, uint3(0, 0, 0), format};
    return tex;
}
ResourceCreationInfo DxCudaInteropImpl::create_interop_event() noexcept {
    auto evt = impl->create_interop_event();
    new Event(evt.handle);
    device_stats->create_event(evt.handle);
    return evt;
}

}// namespace lc::validation