#include "stats.h"
#include <luisa/core/logging.h>

namespace luisa::compute {
DeviceStats::DeviceStats() {}
DeviceStats::~DeviceStats() {}
void DeviceStats::create_event(uint64_t handle) {
    std::lock_guard lck{mtx};
    auto &v = _events_stats.try_emplace(handle).first->second;
    v.event_handle = handle;
}
void DeviceStats::destroy_event(uint64_t handle) {
    std::lock_guard lck{mtx};
    _events_stats.erase(handle);
}
void DeviceStats::create_stream(uint64_t handle, StreamTag stream_tag) {
    std::lock_guard lck{mtx};
    auto &v = _stream_stats.try_emplace(handle).first->second;
    v.stream_handle = handle;
    v.stream_tag = stream_tag;
}
void DeviceStats::destroy_stream(uint64_t handle) {
    std::lock_guard lck{mtx};
    _stream_stats.erase(handle);
}
StreamStats &DeviceStats::_get_stream_stats(uint64_t stream_handle) {
    auto iter = _stream_stats.find(stream_handle);
    LUISA_ASSERT(iter != _stream_stats.end());
    return iter->second;
}
EventStats &DeviceStats::_get_event_stats(uint64_t event_handle) {
    auto iter = _events_stats.find(event_handle);
    LUISA_ASSERT(iter != _events_stats.end());
    return iter->second;
}
void DeviceStats::_process_wait(EventStats &event_stats, StreamStats &stream_stats, uint64_t wait_idx) {
    for (auto &kv : event_stats.signaled_stream) {
        auto &stream_scope = kv.second;
        auto &fence_filter = stream_stats.waited_stream_fence.try_emplace(stream_scope->stream_handle, 0).first->second;
        if (kv.first > fence_filter && kv.first <= wait_idx) {
            stream_stats._next_scope_depending_scopes.emplace_back(stream_scope);
            fence_filter = kv.first;
        }
    }
}

void DeviceStats::signal_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index) {
    std::lock_guard lck{mtx};
    auto &evt = _get_event_stats(event_handle);
    auto &stream = _get_stream_stats(stream_handle);
    if (fence_index <= evt.latest_index) [[unlikely]] {
        LUISA_ERROR("Can not signal a fence {} less than or equal to last fence {}", fence_index, evt.latest_index);
    }
    evt.latest_index = fence_index;
    evt.signaled_stream.emplace_back(fence_index, (!stream.stream_scopes.empty()) ? stream.stream_scopes.back() : luisa::shared_ptr<StreamStatsScope>{});
    while (!evt.wait_stream.empty()) {
        auto &cmd = evt.wait_stream.front();
        if (cmd.fence_idx <= fence_index) {
            _process_wait(_get_event_stats(cmd.event_handle), _get_stream_stats(cmd.stream_handle), cmd.fence_idx);
            evt.wait_stream.pop();
        } else {
            break;
        }
    }
}
void DeviceStats::wait_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index) {
    std::lock_guard lck{mtx};
    auto &evt = _get_event_stats(event_handle);
    auto &stream = _get_stream_stats(stream_handle);
    if (evt.latest_index >= fence_index) {
        _process_wait(evt, stream, fence_index);
    } else {
        evt.wait_stream.emplace(EventStats::WaitCmd{
            .stream_handle = stream.stream_handle,
            .event_handle = evt.event_handle,
            .fence_idx = fence_index});
    }
}
luisa::move_only_function<void()> DeviceStats::dispatch_stream(uint64_t stream_handle, luisa::string &&dispatch_name) {
    std::lock_guard lck{mtx};
    auto &stream = _get_stream_stats(stream_handle);
    auto new_scope = luisa::make_shared<StreamStatsScope>();
    new_scope->stream_handle = stream_handle;
    new_scope->name = std::move(dispatch_name);
    if (!stream._next_scope_depending_scopes.empty()) {
        new_scope->depending_scopes = std::move(stream._next_scope_depending_scopes);
    }
    _unfinished_stage++;
    luisa::move_only_function<void()> func([this, new_scope]() {
        std::lock_guard lck{mtx};
        new_scope->finished_time = clk.toc();
        --_unfinished_stage;
    });
    new_scope->finished_time = 0.;
    auto &v = stream.stream_scopes.emplace_back(std::move(new_scope));
    if (!clk_ticked) {
        clk_ticked = true;
        clk.tic();
        v->start_time = 0;
    } else {
        v->start_time = clk.toc();
    }
    return func;
}
void DeviceStats::finalize() {
    _wait_frame();
    std::lock_guard lck{mtx};
    for (auto &i : _stream_stats) {
        for (auto &scope : i.second.stream_scopes) {
            for (auto &dep : scope->depending_scopes) {
                scope->start_time = std::max(scope->start_time, dep->finished_time);
            }
        }
    }
}
void DeviceStats::_wait_frame() {
    while (_unfinished_stage != 0) {
        std::this_thread::yield();
    }
}

void DeviceStats::reset_frame() {
    _wait_frame();
    std::lock_guard lck{mtx};
    for (auto &i : _stream_stats) {
        i.second.waited_stream_fence.clear();
        i.second.stream_scopes.clear();
        i.second._next_scope_depending_scopes.clear();
    }
    for (auto &i : _events_stats) {
        i.second.signaled_stream.clear();
#ifdef LUISA_USE_SYSTEM_STL
        while (!i.second.wait_stream.empty()) {
            i.second.wait_stream.pop();
        }
#else
        i.second.wait_stream.get_container().clear();
#endif
    }
    clk_ticked = false;
}
}// namespace luisa::compute