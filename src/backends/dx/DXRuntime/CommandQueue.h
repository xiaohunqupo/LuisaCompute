#pragma once
#include <DXRuntime/Device.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <DXRuntime/DxPtr.h>
#include <dxgi1_4.h>
namespace lc::dx {
class CommandBuffer;
class CommandAllocator;
class GpuAllocator;
class LCEvent;
class CommandQueue : vstd::IOperatorNewBase {
public:
    using AllocatorPtr = vstd::unique_ptr<CommandAllocator>;

private:
    struct WaitFence {
    };
    struct CallbackEvent {
        using Variant = vstd::variant<
            AllocatorPtr,
            vstd::vector<vstd::function<void()>>,
            LCEvent const *,
            WaitFence>;
        Variant evt;
        uint64_t fence;
        bool wakeupThread;
        template<typename Arg>
            requires(luisa::is_constructible_v<Variant, Arg &&>)
        CallbackEvent(Arg &&arg,
                      uint64_t fence,
                      bool wakeupThread)
            : evt{std::forward<Arg>(arg)}, fence{fence}, wakeupThread{wakeupThread} {}
    };
    std::atomic_bool enabled = true;
    std::atomic_uint64_t executedFrame = 0;
    std::atomic_uint64_t lastFrame = 0;
    Device *device;
    GpuAllocator *resourceAllocator;
    D3D12_COMMAND_LIST_TYPE type;
    luisa::spin_mutex mtx;
    DxPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> cmdFence;
    vstd::LockFreeArrayQueue<AllocatorPtr> allocatorPool;
    vstd::SingleThreadArrayQueue<CallbackEvent> executedAllocators;
    void ExecuteThread();

public:
    void WaitFrame(uint64 lastFrame);
    uint64 LastFrame() const { return lastFrame; }
    ID3D12CommandQueue *Queue() const { return queue; }
    CommandQueue(
        Device *device,
        GpuAllocator *resourceAllocator,
        D3D12_COMMAND_LIST_TYPE type);
    ~CommandQueue();
    AllocatorPtr CreateAllocator(size_t maxAllocCount);
    void AddEvent(LCEvent const *evt, uint64_t fenceIdx);
    void Signal();
    void Execute(AllocatorPtr &&alloc, vstd::vector<vstd::function<void()>> &&callbacks, luisa::span<std::pair<IDXGISwapChain *, bool>> swapChains, bool cmdlist_is_empty);
    void Complete(uint64 fence);
    void Complete();
    void ForceSync(
        AllocatorPtr &alloc,
        CommandBuffer &cb);
    KILL_MOVE_CONSTRUCT(CommandQueue)
    KILL_COPY_CONSTRUCT(CommandQueue)
private:
    // make sure thread always construct after all members
    std::thread thd;
};
}// namespace lc::dx
