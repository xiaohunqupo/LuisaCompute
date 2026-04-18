#pragma once

#include <volk.h>
#include <luisa/core/spin_mutex.h>
#include "resource.h"

namespace lc::vk {

class Stream;

class Event : public Resource {

    friend class Stream;
    VkSemaphore _semaphore{};
    mutable std::atomic_uint64_t _signaled_event = 0;
    mutable std::atomic_uint64_t _finished_event = 0;
    mutable luisa::spin_mutex _event_mtx;
    mutable uint64_t _last_fence = 0;
    void _update_fence(uint64_t value);
    void _signal(Stream &stream, uint64_t value, VkCommandBuffer *cmdbuffer = nullptr);
    void _signal_sparse(Stream &stream, uint64_t const *value_ptr, VkBindSparseInfo *sparse_info, VkTimelineSemaphoreSubmitInfo *timeline_ptr);
    void _wait(Stream &stream, uint64_t value);
    void _host_wait(uint64_t value);
    void _notify(uint64_t value);

public:
    static VkTimelineSemaphoreSubmitInfo get_timeline_submit(uint64_t const *value_ptr);
    [[nodiscard]] auto semaphore() const { return _semaphore; }
    [[nodiscard]] auto last_fence() const { return _last_fence; }
    [[nodiscard]] bool is_complete(uint64_t fence) const {
        std::lock_guard lck{_event_mtx};
        return _finished_event >= fence;
    }
    void mark_signal_fence(uint64_t fence);
    void sync(uint64_t value);
    Event(Device *device);
    ~Event();
};

}// namespace lc::vk
