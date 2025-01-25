#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

// Note: this instruction must be the terminator of a basic block.
class LC_XIR_API ContinueInst final : public DerivedBranchInstruction<DerivedInstructionTag::CONTINUE> {
public:
    using DerivedBranchInstruction::DerivedBranchInstruction;
    [[nodiscard]] ContinueInst *clone(InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
