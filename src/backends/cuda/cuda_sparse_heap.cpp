#include "cuda_sparse_heap.h"
#include "cuda_device.h"
namespace luisa::compute::cuda {
CUDASparseHeap::CUDASparseHeap(CUDADevice *device, size_t size_bytes) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device->handle().index();
    const auto granularity = device->sparse_granularity();
    size_bytes = ((size_bytes - 1) / granularity + 1) * granularity;
    _size_bytes = size_bytes;
    if (cuMemCreate(&_alloc_handle, _size_bytes, &prop, 0) != CUDA_SUCCESS) {
        LUISA_ERROR("CUDA allocate sparse memory failed.");
    }
}
CUDASparseHeap::~CUDASparseHeap() {
    if (cuMemRelease(_alloc_handle) != CUDA_SUCCESS) {
        LUISA_ERROR("CUDA free sparse memory failed.");
    }
}
}// namespace luisa::compute::cuda