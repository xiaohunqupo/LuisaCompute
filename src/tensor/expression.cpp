#include <luisa/tensor/expression.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/tensor/kernel.h>
namespace luisa::compute {
void TensorExpr::remove_self() noexcept {
    if (_last != nullptr) {
        _last->_next = _next;
        _last = nullptr;
    }
    if (_next != nullptr) {
        _next->_last = _last;
        _next = nullptr;
    }
}
void TensorExpr::add_after(TensorExpr *expr) noexcept {
    remove_self();
    _next = expr->_next;
    _last = expr;
    expr->_next = this;
    if (_next)
        _next->_last = this;
}
void TensorExpr::add_before(TensorExpr *expr) noexcept {
    remove_self();
    _next = expr;
    _last = expr->_last;
    expr->_last = this;
    if (_last)
        _last->_next = this;
}
SetValueExpr::SetValueExpr(
    TensorDataView tensor_data,
    uint32_t value) noexcept
    : tensor_data(tensor_data), value(value) {
}
FullyConnectTensorExpr::FullyConnectTensorExpr(
    TensorDataView const &input_tensor,
    TensorDataView const &output_tensor,
    TensorDataView const &weight_tensor,
    FusedActivation const &fused_activation) noexcept
    : input_tensor(input_tensor),
      output_tensor(output_tensor),
      weight_tensor(weight_tensor),
      fused_activation(fused_activation) {}
ConvolutionExpr::ConvolutionExpr(
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
    luisa::span<uint const> output_paddings,
    uint group_count,
    FusedActivation const &fused_activation) noexcept
    : input_tensor(input_tensor),
      filter_tensor(filter_tensor),
      bias_tensor(bias_tensor),
      output_tensor(output_tensor),
      is_cross_convolution(is_cross_convolution),
      is_backward(is_backward),
      dimension_count(dimension_count),
      group_count(group_count),
      fused_activation(fused_activation) {
    auto builder = TensorBuilder::get_thd_local();
    this->strides = builder->allocate_array<uint>(dimension_count);
    this->dilations = builder->allocate_array<uint>(dimension_count);
    this->output_paddings = builder->allocate_array<uint>(dimension_count);
    this->paddings = builder->allocate_array<uint2>(dimension_count);
    LUISA_DEBUG_ASSERT(strides.size() == dimension_count);
    LUISA_DEBUG_ASSERT(dilations.size() == dimension_count);
    LUISA_DEBUG_ASSERT(paddings.size() == dimension_count);
    LUISA_DEBUG_ASSERT(output_paddings.size() == dimension_count);
    std::memcpy(this->strides, strides.data(), strides.size_bytes());
    std::memcpy(this->dilations, dilations.data(), dilations.size_bytes());
    std::memcpy(this->paddings, paddings.data(), paddings.size_bytes());
    std::memcpy(this->output_paddings, output_paddings.data(), output_paddings.size_bytes());
}
MaxPoolExpr::MaxPoolExpr(
    TensorDataView const &input_tensor,
    TensorDataView const &output_tensor,
    uint dimension_count,
    luisa::span<uint const> strides,
    luisa::span<uint const> window_size,
    luisa::span<uint const> paddings) noexcept
    : input_tensor(input_tensor),
      output_tensor(output_tensor),
      dimension_count(dimension_count) {
    auto builder = TensorBuilder::get_thd_local();
    this->strides = builder->allocate_array<uint>(dimension_count);
    this->window_size = builder->allocate_array<uint>(dimension_count);
    this->paddings = builder->allocate_array<uint2>(dimension_count);

    LUISA_DEBUG_ASSERT(strides.size() == dimension_count);
    LUISA_DEBUG_ASSERT(window_size.size() == dimension_count);
    LUISA_DEBUG_ASSERT(paddings.size() == dimension_count);
    std::memcpy(this->strides, strides.data(), strides.size_bytes());
    std::memcpy(this->window_size, window_size.data(), window_size.size_bytes());
    std::memcpy(this->paddings, paddings.data(), paddings.size_bytes());
}
}// namespace luisa::compute
