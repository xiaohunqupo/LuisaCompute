#pragma once
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/clock.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/queue.h>
namespace luisa::compute {
struct StreamStatsScope;
// struct CommandStatsScope {
//     luisa::vector<luisa::unique_ptr<CommandStatsScope>> nested_scopes;
//     luisa::string scope_name;
//     uint64_t start_command_idx;
//     uint64_t end_command_idx;
//     luisa::variant<StreamStatsScope *, CommandStatsScope *> parent;
// };

struct StreamStatsScope {
    uint64_t stream_handle;
    double start_time;
    double finished_time;
    luisa::string name;
    // luisa::vector<luisa::unique_ptr<CommandStatsScope>> command_scopes;
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> depending_scopes;
};
struct StreamStats {
    uint64_t stream_handle;
    luisa::unordered_map<uint64_t /*stream handle*/, uint64_t /*fence index*/> waited_stream_fence;
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> stream_scopes;
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> next_scope_depending_scopes;
};
struct EventStats {
    struct WaitCmd {
        uint64_t stream_handle;
        uint64_t event_handle;
        uint64_t fence_idx;
    };
    uint64_t event_handle;
    luisa::vector<std::pair<uint64_t, luisa::shared_ptr<StreamStatsScope>>> signaled_stream;
    luisa::queue<WaitCmd> wait_stream;
    uint64_t latest_index;
};

struct DeviceStats {
private:
    luisa::unordered_map<uint64_t, StreamStats> _stream_stats;
    luisa::unordered_map<uint64_t, EventStats> _events_stats;
    Clock clk;
    bool clk_ticked = false;
    std::mutex mtx;// stats is single thread, use global mutex
    std::atomic_uint64_t _unfinished_stage{};
    StreamStats &_get_stream_stats(uint64_t stream_handle);
    EventStats &_get_event_stats(uint64_t event_handle);
    static void _process_wait(EventStats& event_stats, StreamStats& stream_stats, uint64_t wait_idx);
    
public:
    DeviceStats();
    ~DeviceStats();
    void reset_frame();
    void create_event(uint64_t handle);
    void destroy_event(uint64_t handle);
    void create_stream(uint64_t handle);
    void destroy_stream(uint64_t handle);
    void signal_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index);
    void wait_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index);
    void dispatch_stream(uint64_t stream_handle, luisa::string&& dispatch_name, CommandList& cmdlist);
    void wait_frame();
};
}// namespace luisa::compute