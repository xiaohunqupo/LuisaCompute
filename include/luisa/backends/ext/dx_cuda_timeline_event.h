#pragma once
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/runtime/event.h>
#include <luisa/vstl/meta_lib.h>

namespace luisa::compute {
class DxCudaInterop;
namespace dx_cuda_interop {
struct Signal {
    DxCudaInterop *ext;
    void *native_handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
struct Wait {
    DxCudaInterop *ext;
    void *native_handle;
    uint64_t fence;
    void operator()(DeviceInterface *device, uint64_t stream_handle) const && noexcept;
};
}// namespace dx_cuda_interop
class DxCudaTimelineEvent final {
    DxCudaInterop *_ext;

public:
    TimelineEvent dx_event;

private:
    void *_cuda_event;

public:
    [[nodiscard]] auto cuda_event() const noexcept { return _cuda_event; }
    DxCudaTimelineEvent() noexcept : _cuda_event{nullptr} {}
    DxCudaTimelineEvent(DxCudaInterop *ext) noexcept;
    ~DxCudaTimelineEvent() noexcept;
    operator bool() const noexcept {
        return _cuda_event != nullptr;
    }
    DxCudaTimelineEvent(DxCudaTimelineEvent const &) = delete;
    DxCudaTimelineEvent(DxCudaTimelineEvent &&rhs) noexcept
        : _ext{rhs._ext},
          dx_event{std::move(rhs.dx_event)},
          _cuda_event{rhs._cuda_event} {
        rhs._cuda_event = nullptr;
        rhs._ext = nullptr;
    }
    DxCudaTimelineEvent &operator=(DxCudaTimelineEvent const &) = delete;
    DxCudaTimelineEvent &operator=(DxCudaTimelineEvent &&rhs) noexcept {
        this->~DxCudaTimelineEvent();
        vstd::construct_at(this, std::move(rhs));
        return *this;
    }
    [[nodiscard]] auto cuda_signal(uint64_t fence) const noexcept {
        return dx_cuda_interop::Signal{
            .ext = _ext,
            .native_handle = _cuda_event,
            .fence = fence};
    }
    [[nodiscard]] auto cuda_wait(uint64_t fence) const noexcept {
        return dx_cuda_interop::Wait{
            .ext = _ext,
            .native_handle = _cuda_event,
            .fence = fence};
    }

    void cuda_signal_external(DxCudaInterop *interop_ext, void *cu_stream_ptr, uint64_t fence) const noexcept;
    void cuda_wait_external(DxCudaInterop *interop_ext, void *cu_stream_ptr, uint64_t fence) const noexcept;

    [[nodiscard]] auto dx_signal(uint64_t fence) const noexcept {
        return dx_event.signal(fence);
    }
    [[nodiscard]] auto dx_wait(uint64_t fence) const noexcept {
        return dx_event.wait(fence);
    }
};
}// namespace luisa::compute