#pragma once
#include <luisa/tensor/tensor.h>
namespace luisa::compute {
#define LUISA_COMPUTE_TENSOR_EXPRRESSIONS \
    AllocateTensorExpr,                   \
        DeAllocateTensorExpr,             \
        FullyConnectTensorExpr,           \
        ConvolutionExpr,                  \
        MaxPoolExpr

#define LUISA_MAKE_TENSOR_EXPR_DECL(CMD) class CMD;
LUISA_MAP(LUISA_MAKE_TENSOR_EXPR_DECL, LUISA_COMPUTE_TENSOR_EXPRRESSIONS)
#undef LUISA_MAKE_TENSOR_EXPR_DECL

struct TensorExprVisitor {
#define LUISA_MAKE_TENSOR_EXPR_VISITOR_INTERFACE(CMD) \
    virtual void visit(const CMD *) noexcept = 0;
    LUISA_MAP(LUISA_MAKE_TENSOR_EXPR_VISITOR_INTERFACE, LUISA_COMPUTE_TENSOR_EXPRRESSIONS)
#undef LUISA_MAKE_TENSOR_EXPR_VISITOR_INTERFACE
    virtual ~TensorExprVisitor() noexcept = default;
};

class TensorExpr {
public:
    enum struct Tag : uint8_t {
#define LUISA_MAKE_TENSOR_EXPR_TAG(Cmd) E##Cmd,
        LUISA_MAP(LUISA_MAKE_TENSOR_EXPR_TAG, LUISA_COMPUTE_TENSOR_EXPRRESSIONS)
#undef LUISA_MAKE_TENSOR_EXPR_TAG
    };
    [[nodiscard]] virtual Tag tag() noexcept = 0;
    virtual void accept(TensorExprVisitor *) const noexcept = 0;
    virtual ~TensorExpr() noexcept = default;
};

template<typename Derive, TensorExpr::Tag _tag>
class TensorExprCRTPDerive : public TensorExpr {
public:
    Tag tag() noexcept override {
        return _tag;
    }
    static constexpr TensorExpr::Tag const_tag = _tag;
    virtual ~TensorExprCRTPDerive() noexcept = default;
    void accept(TensorExprVisitor *visitor) const noexcept override;
};

#define LUISA_TENSOR_EXPR_CLASS_INHERIT(ClassName) \
public                                             \
    TensorExprCRTPDerive<ClassName, TensorExpr::Tag::E##ClassName>

class AllocateTensorExpr : LUISA_TENSOR_EXPR_CLASS_INHERIT(AllocateTensorExpr) {
public:
    TensorData *alloc_data;
    AllocateTensorExpr(TensorData *alloc_data) noexcept : alloc_data(alloc_data) {}
};
class DeAllocateTensorExpr : LUISA_TENSOR_EXPR_CLASS_INHERIT(DeAllocateTensorExpr) {
public:
    TensorData *dealloc_data;
    DeAllocateTensorExpr(TensorData *dealloc_data) noexcept : dealloc_data(dealloc_data) {}
};
class FullyConnectTensorExpr : LUISA_TENSOR_EXPR_CLASS_INHERIT(FullyConnectTensorExpr) {
public:
    TensorData *input_tensor;
    TensorData *output_tensor;
    TensorData *weight_tensor;
    FullyConnectTensorExpr(
        TensorData *input_tensor,
        TensorData *output_tensor,
        TensorData *weight_tensor) noexcept
        : input_tensor(input_tensor),
          output_tensor(output_tensor),
          weight_tensor(weight_tensor) {}
};
class ConvolutionExpr : LUISA_TENSOR_EXPR_CLASS_INHERIT(ConvolutionExpr) {
    // TODO
};
class MaxPoolExpr : LUISA_TENSOR_EXPR_CLASS_INHERIT(MaxPoolExpr) {
    // TODO
};
#undef LUISA_TENSOR_EXPR_CLASS_INHERIT
template<typename Derive, TensorExpr::Tag _tag>
inline void TensorExprCRTPDerive<Derive, _tag>::accept(TensorExprVisitor *visitor) const noexcept {
    visitor->visit(static_cast<Derive const *>(this));
}
}// namespace luisa::compute
