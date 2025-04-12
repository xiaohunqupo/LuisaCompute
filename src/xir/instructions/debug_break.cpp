#include <luisa/core/logging.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/debug_break.h>

namespace luisa::compute::xir {

DebugBreakInst::DebugBreakInst(BasicBlock *parent_block, Callback callback,
                               luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{parent_block, nullptr}, _callback{callback} { set_operands(operands); }

DebugBreakInst *DebugBreakInst::clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept {
    auto cloned = b.debug_break(_callback);
    cloned->reserve_operands(operand_count());
    for (auto i = 0u; i < operand_count(); i++) {
        auto cloned_op = resolver.resolve(operand(i));
        cloned->add_operand(cloned_op);
    }
    return cloned;
}

}// namespace luisa::compute::xir
