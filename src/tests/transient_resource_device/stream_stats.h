#pragma once
#include <luisa/runtime/command_list.h>
#include <luisa/runtime/stream.h>
#include <luisa/core/stl/variant.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/queue.h>
namespace luisa::compute {
struct StreamStatsScope;
struct CommandStatsScope {
    luisa::vector<luisa::unique_ptr<CommandStatsScope>> nested_scopes;
    luisa::string scope_name;
    uint64_t start_command_idx;
    uint64_t end_command_idx;
    luisa::variant<StreamStatsScope *, CommandStatsScope *> parent;
};

struct StreamStatsScope {
    uint64_t stream_handle;
    uint64_t timeline;
    luisa::vector<luisa::unique_ptr<CommandStatsScope>> scopes;
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> depended_scope;
};
struct StreamStats {
    uint64_t stream_handle;
    uint64_t timeline_counter;
    luisa::vector<luisa::shared_ptr<StreamStatsScope>> stream_scopes;
};
struct EventStats {
    luisa::queue<std::pair<uint64_t, luisa::shared_ptr<StreamStatsScope>>> signaled_stream;
};

struct DeviceStats {
private:
    luisa::unordered_map<uint64_t, StreamStats> _stream_stats;
    luisa::unordered_map<uint64_t, EventStats> _events_stats;

public:
};
}// namespace luisa::compute