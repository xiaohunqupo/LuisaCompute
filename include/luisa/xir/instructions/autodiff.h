#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API AutodiffScope final : public DerivedAutodiffInstruction<AutodiffScope, DerivedInstructionTag::AUTO_DIFF> {

private:
    // TODO: this should be placed into operands
    BasicBlock* _entry_block{nullptr};

public:
    AutodiffScope() = default;
    void set_entry_block(BasicBlock* block) noexcept { _entry_block = block; }
    [[nodiscard]] BasicBlock* entry_block() noexcept { return _entry_block; }
    [[nodiscard]] const BasicBlock* entry_block() const noexcept { return _entry_block; }
    [[nodiscard]] AutodiffScope* clone(InstructionCloneValueResolver& resolver) const noexcept override { return nullptr; }
};

}// namespace luisa::compute::xir
