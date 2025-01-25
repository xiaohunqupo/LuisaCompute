#include <luisa/core/logging.h>
#include <luisa/xir/instructions/arithmetic.h>

namespace luisa::compute::xir {

ArithmeticInst::ArithmeticInst(const Type *type, ArithmeticOp op,
                               luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{type}, InstructionOpMixin{op} {
    set_operands(operands);
}

ArithmeticInst *ArithmeticInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 16u> resolved_operands;
    resolved_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        resolved_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return Pool::current()->create<ArithmeticInst>(type(), op(), resolved_operands);
}

}// namespace luisa::compute::xir
