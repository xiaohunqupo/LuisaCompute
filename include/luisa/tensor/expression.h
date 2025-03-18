#pragma once
#include <luisa/tensor/tensor.h>
#include <luisa/tensor/fused_activation.h>
namespace luisa::compute {
#define LUISA_COMPUTE_TENSOR_EXPRRESSIONS \
    SetValueExpr,                         \
        MultipleTensorExpr,               \
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
    TensorExpr *_last = nullptr;
    TensorExpr *_next = nullptr;
public:
    LC_TENSOR_API void remove_self() noexcept;
    LC_TENSOR_API void add_after(TensorExpr *expr) noexcept;
    LC_TENSOR_API void add_before(TensorExpr *expr) noexcept;

    enum struct Tag : uint8_t {
#define LUISA_MAKE_TENSOR_EXPR_TAG(Cmd) E##Cmd,
        LUISA_MAP(LUISA_MAKE_TENSOR_EXPR_TAG, LUISA_COMPUTE_TENSOR_EXPRRESSIONS)
#undef LUISA_MAKE_TENSOR_EXPR_TAG
    };
    [[nodiscard]] virtual Tag tag() noexcept = 0;
    virtual void accept(TensorExprVisitor *) const noexcept = 0;
    // virtual ~TensorExpr() noexcept = default;
};

template<typename Derive, TensorExpr::Tag _tag>
class TensorExprCRTPDerive : public TensorExpr {
public:
    Tag tag() noexcept override {
        return _tag;
    }
    static constexpr TensorExpr::Tag const_tag = _tag;
    // virtual ~TensorExprCRTPDerive() noexcept = default;
    void accept(TensorExprVisitor *visitor) const noexcept override;
};

#define LUISA_TENSOR_EXPR_CLASS_INHERIT(ClassName) ClassName final : public TensorExprCRTPDerive<ClassName, TensorExpr::Tag::E##ClassName>
class LC_TENSOR_API LUISA_TENSOR_EXPR_CLASS_INHERIT(SetValueExpr) {
public:
    TensorData *tensor_data;
    uint32_t value;
    SetValueExpr(
        TensorData *tensor_data,
        uint32_t value) noexcept;
};
class LC_TENSOR_API LUISA_TENSOR_EXPR_CLASS_INHERIT(MultipleTensorExpr) {
public:
    TensorData *input_tensor;
    TensorData *output_tensor;
    TensorData *weight_tensor;
    FusedActivation fused_activation;
    uint group_count;
    MultipleTensorExpr(
        TensorData *const &input_tensor,
        TensorData *const &output_tensor,
        TensorData *const &weight_tensor,
        FusedActivation const &fused_activation,
        uint group_count) noexcept;
};
class LC_TENSOR_API LUISA_TENSOR_EXPR_CLASS_INHERIT(ConvolutionExpr) {
    TensorData *input_tensor;
    TensorData *filter_tensor;
    TensorData *bias_tensor;
    TensorData *output_tensor;
    bool is_cross_convolution : 1;
    bool is_backward : 1;
    uint dimension_count;
    // dimension count:
    uint *strides;
    uint *dilations;
    uint2 *paddings;// start + end
    uint *output_paddings;
    uint group_count;
    FusedActivation fused_activation;
    ConvolutionExpr(
        TensorData *const &input_tensor,
        TensorData *const &filter_tensor,
        TensorData *const &bias_tensor,
        TensorData *const &output_tensor,
        bool is_cross_convolution,
        bool is_backward,
        uint dimension_count,
        luisa::span<uint const> strides,
        luisa::span<uint const> dilations,
        luisa::span<uint2 const> paddings,
        luisa::span<uint const> output_paddingss,
        uint group_count,
        FusedActivation const &fused_activation) noexcept;
    // TODO
};
class LC_TENSOR_API LUISA_TENSOR_EXPR_CLASS_INHERIT(MaxPoolExpr) {
    TensorData *input_tensor;
    TensorData *output_tensor;
    uint dimension_count;
    // dimension count:
    uint *strides;
    uint *window_size;
    uint2 *paddings;
    MaxPoolExpr(
        TensorData *const &input_tensor,
        TensorData *const &output_tensor,
        uint dimension_count,
        luisa::span<uint const> strides,
        luisa::span<uint const> window_size,
        luisa::span<uint const> paddings) noexcept;
    // TODO
};
#undef LUISA_TENSOR_EXPR_CLASS_INHERIT
template<typename Derive, TensorExpr::Tag _tag>
inline void TensorExprCRTPDerive<Derive, _tag>::accept(TensorExprVisitor *visitor) const noexcept {
    visitor->visit(static_cast<Derive const *>(this));
}
}// namespace luisa::compute
