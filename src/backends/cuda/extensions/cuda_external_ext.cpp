#include "cuda_external_ext.h"
#include "../cuda_device.h"
#include "../cuda_event.h"
#include <cuda.h>
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
void CUDAExternalExtImpl::buffer_copy_async(
    /*CUdeviceptr*/ void *dst_buffer,
    /*CUdeviceptr*/ void *src_buffer,
    size_t size,
    /*CUstream*/ void *stream) {
    device->with_handle([&] {
        LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst_buffer), reinterpret_cast<CUdeviceptr>(src_buffer), size, static_cast<CUstream>(stream)));
    });
}
void CUDAExternalExtImpl::sync_stream(void *cu_stream_ptr) {
    device->with_handle([&] {
        LUISA_CHECK_CUDA(cuStreamSynchronize(static_cast<CUstream>(cu_stream_ptr)));
    });
}
}// namespace luisa::compute::cuda
