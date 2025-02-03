#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API BreakInst final : public DerivedBranchInstruction<BreakInst, DerivedInstructionTag::BREAK> {
public:
    using DerivedBranchInstruction::DerivedBranchInstruction;
    [[nodiscard]] BreakInst *clone(Builder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
