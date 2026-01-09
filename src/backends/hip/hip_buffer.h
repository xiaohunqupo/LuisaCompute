//
// Created by mike on 12/25/25.
//

#pragma once

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

class HIPBuffer {

private:
    hipDeviceptr_t _device_ptr;
    uint64_t _size_bytes : 62;
    uint64_t _is_host : 1;
    uint64_t _is_external : 1;

public:
    struct Binding {
        hipDeviceptr_t ptr;
        uint64_t size_bytes;
    };

public:
    HIPBuffer() noexcept;
    ~HIPBuffer() noexcept;
    [[nodiscard]] static HIPBuffer *create_device_buffer(size_t size_bytes) noexcept;
    [[nodiscard]] static HIPBuffer *create_host_buffer(size_t size_bytes) noexcept;
    [[nodiscard]] static HIPBuffer *import_external_device_buffer(hipDeviceptr_t external_ptr, size_t size_bytes) noexcept;
    [[nodiscard]] static HIPBuffer *import_external_host_buffer(hipDeviceptr_t external_ptr, size_t size_bytes) noexcept;
    static void destroy(HIPBuffer *buffer) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _device_ptr; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
    [[nodiscard]] Binding binding(size_t offset, size_t size) const noexcept;
};

}// namespace luisa::compute::hip
