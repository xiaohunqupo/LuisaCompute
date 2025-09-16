#pragma once

#include <cuda.h>
namespace luisa::compute::cuda {
class CUDADevice;
class CUDASparseHeap {

private:
    CUmemGenericAllocationHandle _alloc_handle{};
    size_t _size_bytes{};

public:
    explicit CUDASparseHeap(CUDADevice *device, size_t size_bytes);
    [[nodiscard]] auto handle() const { return _alloc_handle; }
    ~CUDASparseHeap();
};
}// namespace luisa::compute::cuda