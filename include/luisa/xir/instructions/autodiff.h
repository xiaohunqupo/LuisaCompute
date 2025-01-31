#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API AutodiffScope final : public DerivedAutodiffInstruction<DerivedInstructionTag::AUTO_DIFF> {

private:
    BasicBlock* _entry_block{nullptr};
public:
    AutodiffScope();
    void set_entry_block(BasicBlock* block) noexcept;
    [[nodiscard]] BasicBlock* entry_block() noexcept;
    [[nodiscard]] const BasicBlock* entry_block() const noexcept;
    [[nodiscard]] AutodiffScope* clone(InstructionCloneValueResolver& resolver) const noexcept override;
};

}// namespace luisa::compute::xir
