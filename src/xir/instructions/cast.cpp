#include <luisa/core/logging.h>
#include <luisa/xir/instructions/cast.h>

namespace luisa::compute::xir {

CastInst::CastInst(const Type *target_type, CastOp op, Value *value) noexcept
    : DerivedInstruction{target_type}, InstructionOpMixin{op} {
    auto operands = std::array{value};
    set_operands(operands);
}

Value *CastInst::value() noexcept {
    return operand(0);
}

const Value *CastInst::value() const noexcept {
    return operand(0);
}

void CastInst::set_value(Value *value) noexcept {
    set_operand(0, value);
}

CastInst *CastInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_value = resolver.resolve(value());
    return Pool::current()->create<CastInst>(type(), op(), resolved_value);
}

}// namespace luisa::compute::xir
