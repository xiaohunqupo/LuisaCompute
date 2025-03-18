#pragma once
#include <luisa/tensor/tensor.h>
#include <luisa/tensor/fused_activation.h>
namespace luisa::compute {
#define LUISA_COMPUTE_TENSOR_EXPRRESSIONS \
    SetValueExpr,                         \
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
struct TensorDataView {
    TensorData *data;
    size_t offset;
    size_t size;
};
class LC_TENSOR_API LUISA_TENSOR_EXPR_CLASS_INHERIT(SetValueExpr) {
public:
    TensorDataView tensor_data;
    uint32_t value;
    SetValueExpr(
        TensorDataView tensor_data,
        uint32_t value) noexcept;
};
class LC_TENSOR_API LUISA_TENSOR_EXPR_CLASS_INHERIT(FullyConnectTensorExpr) {
public:
    TensorDataView input_tensor;
    TensorDataView output_tensor;
    TensorDataView weight_tensor;
    FusedActivation fused_activation;
    FullyConnectTensorExpr(
        TensorDataView const &input_tensor,
        TensorDataView const &output_tensor,
        TensorDataView const &weight_tensor,
        FusedActivation const &fused_activation) noexcept;
};
class LC_TENSOR_API LUISA_TENSOR_EXPR_CLASS_INHERIT(ConvolutionExpr) {
    TensorDataView input_tensor;
    TensorDataView filter_tensor;
    TensorDataView bias_tensor;
    TensorDataView output_tensor;
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
        TensorDataView const &input_tensor,
        TensorDataView const &filter_tensor,
        TensorDataView const &bias_tensor,
        TensorDataView const &output_tensor,
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
    TensorDataView input_tensor;
    TensorDataView output_tensor;
    uint dimension_count;
    // dimension count:
    uint *strides;
    uint *window_size;
    uint2 *paddings;
    MaxPoolExpr(
        TensorDataView const &input_tensor,
        TensorDataView const &output_tensor,
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
