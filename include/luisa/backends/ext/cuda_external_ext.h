#pragma once
#include <luisa/runtime/rhi/device_interface.h>
namespace luisa::compute {
class CUDAExternalExt : public DeviceExtension {
public:
    static constexpr luisa::string_view name = "CUDAExternalExt";

    virtual void cuda_stream_signal(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) = 0;
    virtual void cuda_stream_wait(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) = 0;
};
}// namespace luisa::computes