#pragma once
#include <luisa/tensor/tensor.h>
#include <luisa/tensor/expression.h>

namespace luisa::compute {
class LC_TENSOR_API TensorBuilder {
    friend class Tensor;

    luisa::Pool<TensorData, false, false> _tensor_pool;
    luisa::vector<TensorData *> _allocated_tensor;
    luisa::vector<TensorExpr *> _tensor_expr;
    struct Stack {
        void *ptr;
        size_t size;
        size_t offset;
    };
    luisa::vector<Stack> _stack_allocator;
    size_t _allocated_absolute_size = 0;
    void deallocate_tensor(TensorData *tensor) noexcept;

public:
    [[nodiscard]] luisa::span<TensorData *const> allocated_tensor() const noexcept {
        return {_allocated_tensor};
    }
    [[nodiscard]] luisa::span<TensorExpr *const> tensor_expr() const noexcept {
        return {_tensor_expr};
    }
    static void set_thd_local(TensorBuilder *builder);
    static TensorBuilder *get_thd_local();
    void *allocate_stack(size_t size_bytes, size_t alignment = 8) noexcept;
    template<typename T>
        requires(std::is_trivially_destructible_v<T>)
    T *allocate_array(size_t size) {
        return std::launder(reinterpret_cast<T *>(allocate_stack(size * sizeof(T), alignof(T))));
    }
    TensorData *allocate_tensor(
        luisa::span<uint32_t const> sizes,
        TensorElementType element_type) noexcept;
    template<typename T, typename... Args>
        requires(std::is_base_of_v<TensorExpr, T> && std::is_trivially_destructible_v<T> && luisa::is_constructible_v<T, uint64_t, Args && ...>)
    T *allocate_expr(Args &&...args) {
        auto ptr = allocate_stack(sizeof(T), alignof(T));
        auto r = new (ptr) T{_tensor_expr.size(), std::forward<Args>(args)...};
        _tensor_expr.emplace_back(r);
        return r;
    }
    TensorBuilder() noexcept;
    ~TensorBuilder() noexcept;
    TensorBuilder(TensorBuilder const &) = delete;
    TensorBuilder(TensorBuilder &&) = delete;
};
}// namespace luisa::compute