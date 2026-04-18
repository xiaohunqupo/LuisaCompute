#pragma once
#include <Resource/Resource.h>
namespace lc::dx {
enum class CmdQueueTag {
    MainCmd,
    DStorage,
};
class CmdQueueBase : public Resource {
protected:
    CmdQueueTag _tag;
    CmdQueueBase(Device *device, CmdQueueTag tag);
    ~CmdQueueBase() = default;

public:
    luisa::function<void(luisa::string_view)> log_callback;
    CmdQueueTag tag() const { return _tag; }
    Resource::Tag get_tag() const override {
        return Resource::Tag::CommandQueue;
    }
};
}// namespace lc::dx
