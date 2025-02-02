#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API AutodiffScopeInst final : public ControlFlowMergeMixin<DerivedTerminatorInstruction<AutodiffScopeInst, DerivedInstructionTag::AUTODIFF>> {

public:
    static constexpr size_t operand_index_entry_block = 0u;

public:
    AutodiffScopeInst() noexcept;
    void set_entry_block(BasicBlock *block) noexcept;
    BasicBlock *create_entry_block(bool overwrite_existing = false) noexcept;
    [[nodiscard]] BasicBlock *entry_block() noexcept;
    [[nodiscard]] const BasicBlock *entry_block() const noexcept;
    [[nodiscard]] AutodiffScopeInst *clone(InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
