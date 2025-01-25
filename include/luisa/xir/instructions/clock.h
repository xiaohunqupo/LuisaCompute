#pragma once

#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

class LC_XIR_API ClockInst final : public DerivedInstruction<DerivedInstructionTag::CLOCK> {
public:
    ClockInst() noexcept;
    [[nodiscard]] ClockInst *clone(InstructionCloneValueResolver &resolver) const noexcept override;
};

}// namespace luisa::compute::xir
