//
// Created by mike on 1/10/26.
//

#pragma once

#include <hip/hip_runtime.h>

namespace luisa::compute::hip {

class HIPDevice;

// timeline semaphore style event implemented with stream memory operations
class HIPEvent {

private:
    hipDeviceptr_t _semaphore_device_ptr;
    volatile uint64_t *_semaphore_host_ptr;
    uint64_t _initial_value;

    [[nodiscard]] uint64_t _remap_value(uint64_t value) const noexcept;

public:
    explicit HIPEvent(HIPDevice *device) noexcept;
    ~HIPEvent() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _semaphore_device_ptr; }
    void signal(hipStream_t stream, uint64_t value) noexcept;
    void wait(hipStream_t stream, uint64_t value) noexcept;
    void synchronize(uint64_t value) const noexcept;
    [[nodiscard]] bool has_signaled(uint64_t value) const noexcept;
};

}// namespace luisa::compute::hip
