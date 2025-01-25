#include <luisa/ast/type_registry.h>
#include <luisa/xir/instructions/clock.h>

namespace luisa::compute::xir {

ClockInst::ClockInst() noexcept
    : DerivedInstruction{Type::of<luisa::ulong>()} {}

ClockInst *ClockInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    return Pool::current()->create<ClockInst>();
}

}// namespace luisa::compute::xir
