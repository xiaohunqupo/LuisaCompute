#pragma once
#include <luisa/ast/expression.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
namespace luisa::compute {

template<typename T>
class CoopVector {
private:
    const RefExpr *_expression;
    uint32_t _size;

public:
    explicit CoopVector(uint32_t n) noexcept : _expression{detail::FunctionBuilder::current()->local(Type::cooperative_vector(Type::of<T>(), n))}, _size{n} {}

    CoopVector(CoopVector &&) noexcept = default;
    CoopVector(const CoopVector &) noexcept = delete;
    CoopVector &operator=(CoopVector &&) noexcept = delete;
    CoopVector &operator=(const CoopVector &) noexcept = delete;

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] operator Expr<CoopVector>() const noexcept;
    [[nodiscard]] auto size_bytes() const noexcept { return size_t(_size) * sizeof(T); }

    /// Access at index
    template<typename U>
        requires is_integral_expr_v<U>
    [[nodiscard]] auto &operator[](U &&index) const noexcept;

    /// Read index
    template<typename I>
    [[nodiscard]] auto read(I &&index) const noexcept {
        return (*this)[std::forward<I>(index)];
    }
    /// Write index
    template<typename I, typename U>
    void write(I &&i, U &&u) const noexcept {
        (*this)[std::forward<I>(i)] = std::forward<U>(u);
    }
};

class CoopVectorRef {
private:
    const RefExpr *_expression;
    CoopRefVecType _coop_ref_type;
    uint32_t _size;

public:
    explicit CoopVectorRef(CoopRefVecType coop_ref_type, uint32_t n) noexcept : _expression{detail::FunctionBuilder::current()->local(Type::cooperative_vector_ref(coop_ref_type, n))}, _coop_ref_type{coop_ref_type}, _size{n} {}

    CoopVectorRef(CoopVectorRef &&) noexcept = default;
    CoopVectorRef(const CoopVectorRef &) noexcept = delete;
    CoopVectorRef &operator=(CoopVectorRef &&) noexcept = delete;
    CoopVectorRef &operator=(const CoopVectorRef &) noexcept = delete;

    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept { return size_t(_size) * coop_ref_vec_type_size(_coop_ref_type); }
    [[nodiscard]] Expr<uint> byte_offset() const noexcept;
    [[nodiscard]] operator Expr<uint>() const noexcept;
    void set_byte_offset(uint value) noexcept;
    void set_byte_offset(Expr<uint> value) noexcept;
};
class CoopMatrixRef {
private:
    const RefExpr *_expression;
    CoopRefVecType _coop_ref_type;
    uint2 _size;

public:
    explicit CoopMatrixRef(CoopRefVecType coop_ref_type, uint n, uint m) noexcept : _expression{detail::FunctionBuilder::current()->local(Type::cooperative_matrix_ref(coop_ref_type, n, m))}, _coop_ref_type{coop_ref_type}, _size{n, m} {}

    CoopMatrixRef(CoopMatrixRef &&) noexcept = default;
    CoopMatrixRef(const CoopMatrixRef &) noexcept = delete;
    CoopMatrixRef &operator=(CoopMatrixRef &&) noexcept = delete;
    CoopMatrixRef &operator=(const CoopMatrixRef &) noexcept = delete;
    void set_byte_offset(uint value) noexcept;
    void set_byte_offset(Expr<uint> value) noexcept;
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto size() const noexcept { return _size; }
    [[nodiscard]] auto size_bytes() const noexcept { return size_t(_size.x) * size_t(_size.y) * coop_ref_vec_type_size(_coop_ref_type); }
    [[nodiscard]] Expr<uint> byte_offset() const noexcept;
    [[nodiscard]] operator Expr<uint>() const noexcept;
};

}// namespace luisa::compute
