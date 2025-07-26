#pragma once
#include <DXRuntime/Device.h>
#include <Resource/Resource.h>
namespace lc::dx {
class CommandQueue;
class DStorageCommandQueue;
class LCEvent : public Resource {
public:
    ComPtr<ID3D12Fence> fence;
    mutable std::atomic_uint64_t finishedEvent = 0;
    mutable luisa::spin_mutex eventMtx;
    mutable uint64_t lastFence = 0;
    Tag GetTag() const override { return Tag::Event; }
    ID3D12Fence *Fence() const { return fence.Get(); }
    LCEvent(Device *device, bool shared = false);
    ~LCEvent();
    void Sync(uint64_t fence) const;
    void Signal(CommandQueue *queue, uint64 fenceIdx) const;
    void Signal(DStorageCommandQueue *queue, uint64 fenceIdx) const;
    void Wait(CommandQueue *queue, uint64 fenceIdx) const;
    bool IsComplete(uint64 fenceIdx) const;
};
}// namespace lc::dx
