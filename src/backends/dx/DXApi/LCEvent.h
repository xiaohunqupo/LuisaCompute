#pragma once
#include <DXRuntime/Device.h>
#include <Resource/Resource.h>
namespace lc::dx {
class CommandQueue;
class DStorageCommandQueue;
class LCEvent : public Resource {
public:
    ComPtr<ID3D12Fence> _fence;
    mutable std::atomic_uint64_t finished_event = 0;
    mutable luisa::spin_mutex event_mtx;
    mutable uint64_t last_fence = 0;
    Tag GetTag() const override { return Tag::Event; }
    ID3D12Fence *fence() const { return _fence.Get(); }
    LCEvent(Device *device, bool shared = false);
    ~LCEvent();
    void sync(uint64_t fence) const;
    void signal(CommandQueue *queue, uint64_t fenceIdx) const;
    void signal(DStorageCommandQueue *queue, uint64_t fenceIdx) const;
    void wait(CommandQueue *queue, uint64_t fenceIdx) const;
    bool is_complete(uint64_t fenceIdx) const;
};
}// namespace lc::dx
