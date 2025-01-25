#include <luisa/xir/instructions/assert.h>

namespace luisa::compute::xir {

AssertInst::AssertInst(Value *condition, luisa::string message) noexcept
    : DerivedInstruction{nullptr}, _message{std::move(message)} {
    set_operands(std::array{condition});
}

void AssertInst::set_condition(Value *condition) noexcept {
    set_operand(operand_index_condition, condition);
}

Value *AssertInst::condition() noexcept {
    return operand(operand_index_condition);
}

const Value *AssertInst::condition() const noexcept {
    return operand(operand_index_condition);
}

void AssertInst::set_message(luisa::string_view message) noexcept {
    _message = message;
}

luisa::string_view AssertInst::message() const noexcept {
    return _message;
}

AssertInst *AssertInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_cond = resolver.resolve(condition());
    return Pool::current()->create<AssertInst>(resolved_cond, _message);
}

}// namespace luisa::compute::xir
