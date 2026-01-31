#pragma once

#include <mutex>
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/clock.h>
#include <luisa/backends/ext/stats_ext.h>

namespace luisa::compute {

struct DeviceStats {
private:
    struct DeviceHandle {
        uint64_t device_handle;
        uint64_t resource_handle;
        [[nodiscard]] bool operator=(DeviceHandle const &b) const noexcept {
            return device_handle == b.device_handle && resource_handle == b.resource_handle;
        }
    };
    luisa::unordered_map<uint64_t, StreamStats> _stream_stats;
    luisa::unordered_map<uint64_t, EventStats> _events_stats;
    bool clk_ticked = false;
    std::atomic_uint64_t _unfinished_stage{};
    StreamStats &_get_stream_stats(uint64_t stream_handle);
    EventStats &_get_event_stats(uint64_t event_handle);
    static void _process_wait(EventStats &event_stats, StreamStats &stream_stats, uint64_t wait_idx);
    void _wait_frame();
public:
    DeviceStats();
    ~DeviceStats();
    static DeviceStats *add_ref();
    static void deref();
    luisa::string next_stats_stream_name;
    std::atomic_bool stats_enabled{false};
    std::mutex mtx;// stats is single thread, use global mutex
    Clock clk;
    void reset_frame();
    void create_event(uint64_t handle);
    void destroy_event(uint64_t handle);
    void create_stream(uint64_t handle, StreamTag stream_tag);
    void destroy_stream(uint64_t handle);
    void signal_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index);
    void wait_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index);
    luisa::move_only_function<void()> dispatch_stream(uint64_t stream_handle, luisa::string &&dispatch_name, bool &require_tic, double *&start_time_ptr);
    void finalize();
    auto const &stream_stats() const { return _stream_stats; }
};
}// namespace luisa::compute