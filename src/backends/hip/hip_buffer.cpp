//
// Created by mike on 12/25/25.
//

#include <luisa/core/stl/memory.h>

#include "hip_check.h"
#include "hip_buffer.h"

namespace luisa::compute::hip {

inline HIPBuffer::HIPBuffer() noexcept
    : _device_ptr{}, _size_bytes{}, _is_host{}, _is_external{} {}

HIPBuffer::~HIPBuffer() noexcept {
    if (!_is_external) {
        if (_is_host) {
            LUISA_CHECK_HIP(hipHostFree(_device_ptr));
        } else {
            LUISA_CHECK_HIP(hipFree(_device_ptr));
        }
    }
}

HIPBuffer *HIPBuffer::create_device_buffer(size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    LUISA_CHECK_HIP(hipMalloc(&buffer->_device_ptr, size_bytes));
    buffer->_size_bytes = size_bytes;
    buffer->_is_host = false;
    buffer->_is_external = false;
    return buffer;
}

HIPBuffer *HIPBuffer::create_host_buffer(size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    LUISA_CHECK_HIP(hipHostMalloc(&buffer->_device_ptr, size_bytes));
    buffer->_size_bytes = size_bytes;
    buffer->_is_host = true;
    buffer->_is_external = false;
    return buffer;
}

HIPBuffer *HIPBuffer::import_external_device_buffer(hipDeviceptr_t external_ptr, size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    buffer->_device_ptr = external_ptr;
    buffer->_size_bytes = size_bytes;
    buffer->_is_host = false;
    buffer->_is_external = true;
    return buffer;
}

HIPBuffer *HIPBuffer::import_external_host_buffer(hipDeviceptr_t external_ptr, size_t size_bytes) noexcept {
    auto buffer = luisa::new_with_allocator<HIPBuffer>();
    buffer->_device_ptr = external_ptr;
    buffer->_size_bytes = size_bytes;
    buffer->_is_host = true;
    buffer->_is_external = true;
    return buffer;
}

void HIPBuffer::destroy(HIPBuffer *buffer) noexcept {
    luisa::delete_with_allocator(buffer);
}

}// namespace luisa::compute::hip