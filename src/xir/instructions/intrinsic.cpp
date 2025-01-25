#include <luisa/xir/instructions/intrinsic.h>

namespace luisa::compute::xir {

IntrinsicInst::IntrinsicInst(const Type *type, IntrinsicOp op,
                             luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{type}, InstructionOpMixin{op} { set_operands(operands); }

IntrinsicInst *IntrinsicInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 16u> resolved_operands;
    resolved_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        resolved_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return Pool::current()->create<IntrinsicInst>(type(), op(), resolved_operands);
}

}// namespace luisa::compute::xir
