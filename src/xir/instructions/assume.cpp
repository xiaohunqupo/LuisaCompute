#include <luisa/xir/instructions/assume.h>

namespace luisa::compute::xir {

AssumeInst::AssumeInst(Value *condition, luisa::string message) noexcept
    : DerivedInstruction{nullptr}, _message{std::move(message)} {
    set_operands(std::array{condition});
}

void AssumeInst::set_condition(Value *condition) noexcept {
    set_operand(operand_index_condition, condition);
}

Value *AssumeInst::condition() noexcept {
    return operand(operand_index_condition);
}

const Value *AssumeInst::condition() const noexcept {
    return operand(operand_index_condition);
}

void AssumeInst::set_message(luisa::string_view message) noexcept {
    _message = message;
}

luisa::string_view AssumeInst::message() const noexcept {
    return _message;
}

AssumeInst *AssumeInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    auto resolved_cond = resolver.resolve(condition());
    return Pool::current()->create<AssumeInst>(resolved_cond, _message);
}

}// namespace luisa::compute::xir
