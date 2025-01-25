#include <luisa/xir/instructions/alloca.h>

namespace luisa::compute::xir {

AllocaInst::AllocaInst(const Type *type, AllocSpace space) noexcept
    : DerivedInstruction{type}, _space{space} {}

void AllocaInst::set_space(AllocSpace space) noexcept {
    _space = space;
}

AllocaInst *AllocaInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    return Pool::current()->create<AllocaInst>(type(), space());
}

}// namespace luisa::compute::xir
