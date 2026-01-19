#include "cuda_external_ext.h"
#include "../cuda_device.h"
#include "../cuda_event.h"
namespace luisa::compute::cuda {
void CUDAExternalExtImpl::cuda_stream_signal(
    /*CUstream*/ void *cu_stream_ptr,
    uint64_t cuda_event_handle,
    uint64_t fence_index) {
    device->with_handle([&] {
        reinterpret_cast<cuda::CUDAEvent *>(cuda_event_handle)->signal(static_cast<CUstream>(cu_stream_ptr), fence_index);
    });
}
void CUDAExternalExtImpl::cuda_stream_wait(
    /*CUstream*/ void *cu_stream_ptr,
    uint64_t cuda_event_handle,
    uint64_t fence_index) {
    device->with_handle([&] {
        reinterpret_cast<cuda::CUDAEvent *>(cuda_event_handle)->wait(static_cast<CUstream>(cu_stream_ptr), fence_index);
    });
}
}// namespace luisa::compute::cuda
