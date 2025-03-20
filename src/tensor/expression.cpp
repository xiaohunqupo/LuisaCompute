#include <luisa/tensor/expression.h>
#include <luisa/tensor/tensor_builder.h>
#include <luisa/tensor/kernel.h>
namespace luisa::compute {
/////////////// Static assert
static_assert(std::is_trivially_destructible_v<TensorData>, "Tensor data must be trivially destructible.");
/////////////// Static assert

// std::pair<TensorExpr *, TensorExpr *> TensorExpr::remove_self() noexcept {
//     std::pair<TensorExpr *, TensorExpr *> r{_last, _next};
//     if (_last != nullptr) {
//         _last->_next = _next;
//         _last = nullptr;
//     }
//     if (_next != nullptr) {
//         _next->_last = _last;
//         _next = nullptr;
//     }
//     return r;
// }
// void TensorExpr::add_after(TensorExpr *expr) noexcept {
//     remove_self();
//     _next = expr->_next;
//     _last = expr;
//     expr->_next = this;
//     if (_next)
//         _next->_last = this;
// }
// void TensorExpr::add_before(TensorExpr *expr) noexcept {
//     remove_self();
//     _next = expr;
//     _last = expr->_last;
//     expr->_last = this;
//     if (_last)
//         _last->_next = this;
// }
ScopeExpr::ScopeExpr(uint64_t idx) noexcept : BaseClass(idx) {
}
ScopeExpr::~ScopeExpr() noexcept {
    for (auto &i : expressions) {
        std::destroy_at(i);
    }
}
SetValueExpr::SetValueExpr(
    uint64_t idx,
    TensorData *tensor_data,
    uint32_t value) noexcept
    : BaseClass(idx), tensor_data(tensor_data), value(value) {
}
MultipleTensorExpr::MultipleTensorExpr(
    uint64_t idx,
    TensorData *input_tensor,
    TensorData *output_tensor,
    TensorData *weight_tensor,
    FusedActivation const &fused_activation,
    uint group_count) noexcept
    : BaseClass(idx),
      input_tensor(input_tensor),
      output_tensor(output_tensor),
      weight_tensor(weight_tensor),
      fused_activation(fused_activation),
      group_count(group_count) {
}
TestExpr::TestExpr(
    uint64_t idx,
    TensorData *input,
    TensorData *output,
    luisa::string_view name) noexcept
    : BaseClass(idx),
      input(input),
      output(output),
      name(name) {
}
}// namespace luisa::compute
