#include <luisa/xir/instructions/load.h>

namespace luisa::compute::xir {

LoadInst::LoadInst(const Type *type, Value *variable) noexcept
    : DerivedInstruction{type} {
    auto operands = std::array{variable};
    set_operands(operands);
}

Value *LoadInst::variable() noexcept {
    return operand(0);
}

const Value *LoadInst::variable() const noexcept {
    return operand(0);
}

void LoadInst::set_variable(Value *variable) noexcept {
    return set_operand(0, variable);
}

LoadInst *LoadInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_variable = resolver.resolve(variable());
    return Pool::current()->create<LoadInst>(type(), resolved_variable);
}

}// namespace luisa::compute::xir
