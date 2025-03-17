#pragma once
#include <luisa/tensor/tensor.h>

namespace luisa::compute {
class LC_TENSOR_API TensorBuilder {
    friend class Tensor;

    luisa::Pool<TensorData, false, false> _tensor_pool;
    luisa::vector<TensorData *> _allocated_tensor;
    size_t _allocated_absolute_size = 0;
    uint64_t _id_counter = 0;
    void deallocate_tensor(TensorData *tensor) noexcept;

public:
    TensorData *allocate_tensor(
        luisa::span<size_t const> sizes,
        TensorElementType element_type) noexcept;
    TensorBuilder() noexcept;
    ~TensorBuilder() noexcept;
    TensorBuilder(TensorBuilder const &) = delete;
    TensorBuilder(TensorBuilder &&) = delete;
};
}// namespace luisa::compute