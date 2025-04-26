#pragma once
#include <luisa/tensor/tensor.h>
#include <luisa/tensor/expression.h>

namespace luisa::compute {
class LC_TENSOR_API TensorBuilder {
    friend class Tensor;

    luisa::Pool<TensorData, false, false> _tensor_pool;
    luisa::vector<TensorData *> _allocated_tensor;
    luisa::vector<TensorData *> _arguments;
    luisa::vector<std::pair<TensorData *, Argument::Buffer>> _captured_arguments;
    luisa::vector<TensorData *> _outputs;
    luisa::vector<ScopeExpr *> _expr_stack;
    struct Stack {
        luisa::unique_ptr<std::byte> ptr;
        uint64_t size;
        uint64_t offset;
        Stack(
            std::byte *ptr,
            uint64_t size,
            uint64_t offset) noexcept
            : ptr(ptr), size(size), offset(offset) {}
    };
    luisa::vector<Stack> _stack_allocator;
    ScopeExpr _root_expr;
    uint64_t _allocated_absolute_size = 0;
    void deallocate_tensor(TensorData *tensor) noexcept;

public:
    auto &root_expr() noexcept { return _root_expr; }
    auto const &root_expr() const noexcept { return _root_expr; }
    [[nodiscard]] luisa::span<TensorData *const> allocated_tensor() const noexcept {
        return {_allocated_tensor};
    }
    [[nodiscard]] luisa::span<TensorData *const> arguments() const noexcept {
        return {_arguments};
    }
    [[nodiscard]] luisa::span<std::pair<TensorData *, Argument::Buffer> const> captured_arguments() const noexcept {
        return {_captured_arguments};
    }
    [[nodiscard]] luisa::span<TensorData *const> outputs() const noexcept {
        return {_outputs};
    }
    [[nodiscard]] auto current_scope() noexcept { return _expr_stack.back(); }
    static void set_thd_local(TensorBuilder *builder);
    static TensorBuilder *get_thd_local();
    void *allocate_stack(uint64_t size_bytes, uint64_t alignment = 8) noexcept;
    template<typename T>
        requires(std::is_trivially_destructible_v<T>)
    T *allocate_array(uint64_t size) noexcept {
        return std::launder(reinterpret_cast<T *>(allocate_stack(size * sizeof(T), alignof(T))));
    }
    TensorData *allocate_tensor(
        luisa::span<uint64_t const> sizes,
        TensorElementType element_type) noexcept;
    void push_argument(TensorData *data) noexcept {
        _arguments.emplace_back(data);
    }
    void push_captured_arguments(TensorData *data, Argument::Buffer buffer) noexcept {
        _captured_arguments.emplace_back(data, buffer);
    }
    void push_output(TensorData *data) noexcept {
        _outputs.emplace_back(data);
    }
    void push_scope() noexcept;
    void pop_scope() noexcept;
    TensorBuilder() noexcept;
    ~TensorBuilder() noexcept;
    TensorBuilder(TensorBuilder const &) = delete;
    TensorBuilder(TensorBuilder &&) = delete;
};
template<typename T, typename... Args>
    requires(std::is_base_of_v<TensorExpr, T> && luisa::is_constructible_v<T, uint64_t, Args && ...>)
inline T *ScopeExpr::allocate_expr(Args &&...args) noexcept {
    auto builder = TensorBuilder::get_thd_local();
    auto ptr = static_cast<T *>(builder->allocate_stack(sizeof(T), alignof(T)));
    std::construct_at(ptr, expressions.size(), std::forward<Args>(args)...);
    expressions.emplace_back(ptr);
    return ptr;
}
}// namespace luisa::compute