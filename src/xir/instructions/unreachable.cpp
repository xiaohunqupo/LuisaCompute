#include <luisa/xir/instructions/unreachable.h>

namespace luisa::compute::xir {

UnreachableInst::UnreachableInst(luisa::string message) noexcept
    : _message{std::move(message)} {}

void UnreachableInst::set_message(luisa::string_view message) noexcept {
    _message = message;
}

luisa::string_view UnreachableInst::message() const noexcept {
    return _message;
}

UnreachableInst *UnreachableInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    return Pool::current()->create<UnreachableInst>(_message);
}

}// namespace luisa::compute::xir
