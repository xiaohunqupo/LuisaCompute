#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

enum struct IntrinsicOp {

    // no-op placeholder
    NOP,

    // autodiff ops
    AUTODIFF_REQUIRES_GRADIENT,  // (expr) -> void
    AUTODIFF_GRADIENT,           // (expr) -> expr
    AUTODIFF_GRADIENT_MARKER,    // (ref, expr) -> void
    AUTODIFF_ACCUMULATE_GRADIENT,// (ref, expr) -> void
    AUTODIFF_BACKWARD,           // (expr) -> void
    AUTODIFF_DETACH,             // (expr) -> expr
};

[[nodiscard]] LC_XIR_API luisa::string_view to_string(IntrinsicOp op) noexcept;
[[nodiscard]] LC_XIR_API IntrinsicOp intrinsic_op_from_string(luisa::string_view name) noexcept;

class LC_XIR_API IntrinsicInst final : public DerivedInstruction<DerivedInstructionTag::INTRINSIC>,
                                       public InstructionOpMixin<IntrinsicOp> {
public:
    explicit IntrinsicInst(const Type *type = nullptr,
                           IntrinsicOp op = IntrinsicOp::NOP,
                           luisa::span<Value *const> operands = {}) noexcept;
    [[nodiscard]] IntrinsicInst *clone(InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
