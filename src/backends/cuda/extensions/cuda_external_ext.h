#pragma once
#include <luisa/backends/ext/cuda_external_ext.h>
namespace luisa::compute::cuda {
class CUDADevice;
class CUDAExternalExtImpl : public luisa::compute::CUDAExternalExt {
public:
    CUDADevice *device;
    CUDAExternalExtImpl(CUDADevice *device) : device{device} {}
    void cuda_stream_signal(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) override;
    void cuda_stream_wait(
        /*CUstream*/ void *cu_stream_ptr,
        uint64_t cuda_event_handle,
        uint64_t fence_index) override;
};
}// namespace luisa::compute::cuda