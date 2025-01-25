#include <luisa/xir/instructions/thread_group.h>

namespace luisa::compute::xir {

ThreadGroupInst::ThreadGroupInst(const Type *type, ThreadGroupOp op,
                                 luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{type}, InstructionOpMixin{op} {
    if (!operands.empty()) {
        set_operands(operands);
    }
}

ThreadGroupInst *ThreadGroupInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 16u> resolved_operands;
    resolved_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        resolved_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return Pool::current()->create<ThreadGroupInst>(type(), op(), resolved_operands);
}

}// namespace luisa::compute::xir
