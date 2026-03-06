#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/queue.h>
#include <luisa/core/stl/variant.h>

namespace luisa::compute {
struct StreamStatsScope {
    uint64_t stream_handle;
    double start_time;
    double finished_time;
    luisa::string name;
    // luisa::vector<luisa::unique_ptr<CommandStatsScope>> command_scopes;
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> depending_scopes;
};
struct DeviceStats;
struct StreamStats {
    friend struct DeviceStats;
    StreamTag stream_tag;
    uint64_t stream_handle;
    luisa::unordered_map<uint64_t /*stream handle*/, uint64_t /*fence index*/> waited_stream_fence;
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> stream_scopes;
private:
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> _next_scope_depending_scopes;
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
// Working for validation layer
struct StatsExt : public DeviceExtension {
    static constexpr luisa::string_view name = "StatsExt";
    virtual void begin_stats() noexcept = 0;
    virtual void set_next_dispatch_name(luisa::string &&name) noexcept = 0;
    [[nodiscard]] virtual luisa::unordered_map<uint64_t, StreamStats> const &end_stats() noexcept = 0;
    // External
    virtual void registe_external_event(uint64_t handle) noexcept = 0;
    virtual void unregiste_external_event(uint64_t handle) noexcept = 0;
    virtual void registe_external_stream(uint64_t handle, StreamTag stream_tag) noexcept = 0;
    virtual void unregiste_external_stream(uint64_t handle) noexcept = 0;
    virtual void signal_external_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index) noexcept = 0;
    virtual void wait_external_event(uint64_t event_handle, uint64_t stream_handle, uint64_t fence_index) noexcept = 0;
protected:
    ~StatsExt() = default;
};
}// namespace luisa::compute