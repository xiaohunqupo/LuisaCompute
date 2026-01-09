//
// Created by mike on 1/10/26.
//

#include "hip_check.h"
#include "hip_device.h"
#include "hip_event.h"

namespace luisa::compute::hip {

inline uint64_t HIPEvent::_remap_value(uint64_t value) const noexcept {
    return value + _initial_value;
}

HIPEvent::HIPEvent(HIPDevice *device) noexcept
    : _semaphore_device_ptr{}, _semaphore_host_ptr{}, _initial_value{} {
    auto stream_mem_op_support = 0;
    LUISA_CHECK_HIP(hipDeviceGetAttribute(&stream_mem_op_support,
                                          hipDeviceAttributeCanUseStreamWaitValue,
                                          device->device_id()));
    LUISA_ASSERT(stream_mem_op_support,
                 "HIP device does not support stream-ordered memory operations. "
                 "HIPEvent cannot be created.");
    // allocate memory for the semaphore with the hipMallocSignalMemory flag
    LUISA_CHECK_HIP(hipExtMallocWithFlags(
        &_semaphore_device_ptr, sizeof(uint64_t), hipMallocSignalMemory));
    // get host pointer and initialize to zero
    void *host_ptr = nullptr;
    LUISA_CHECK_HIP(hipHostGetDevicePointer(&host_ptr, _semaphore_device_ptr, 0u));
    LUISA_ASSERT(host_ptr != nullptr, "Failed to obtain HIP event semaphore host pointer.");
    _semaphore_host_ptr = static_cast<volatile uint64_t *>(host_ptr);
    _initial_value = *_semaphore_host_ptr;
    LUISA_ASSERT(_initial_value <= std::numeric_limits<int64_t>::max(),
                 "HIP event semaphore host pointer initialized to invalid value: {}.",
                 _initial_value);
    LUISA_VERBOSE_WITH_LOCATION("Created HIPEvent (semaphore device ptr = {}, host ptr = {}, initial value = {}).",
                                _semaphore_device_ptr, static_cast<void *>(const_cast<uint64_t *>(_semaphore_host_ptr)),
                                _initial_value);
}

HIPEvent::~HIPEvent() noexcept {
    if (_semaphore_device_ptr) {
        LUISA_CHECK_HIP(hipHostFree(_semaphore_device_ptr));
    }
}

void HIPEvent::signal(hipStream_t stream, uint64_t value) noexcept {
    LUISA_CHECK_HIP(hipStreamWriteValue64(stream, _semaphore_device_ptr, _remap_value(value), 0u));
}

void HIPEvent::wait(hipStream_t stream, uint64_t value) noexcept {
    LUISA_CHECK_HIP(hipStreamWaitValue64(stream, _semaphore_device_ptr, _remap_value(value), hipStreamWaitValueGte));
}

void HIPEvent::synchronize(uint64_t value) const noexcept {
    constexpr auto max_wait_iterations_before_yield = 1024u;
    auto wait_iterations = 0u;
    while (!has_signaled(value)) {
        if (++wait_iterations >= max_wait_iterations_before_yield) {
            wait_iterations = 0u;
            std::this_thread::yield();
        }
    }
}

bool HIPEvent::has_signaled(uint64_t value) const noexcept {
    return *_semaphore_host_ptr >= _remap_value(value);
}

}// namespace luisa::compute::hip
