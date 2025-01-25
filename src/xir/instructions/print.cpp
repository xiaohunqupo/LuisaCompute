//
// Created by Mike on 2024/10/20.
//

#include <luisa/xir/instructions/print.h>

namespace luisa::compute::xir {

PrintInst::PrintInst(luisa::string format,
                     luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{nullptr},
      _format{std::move(format)} { set_operands(operands); }

PrintInst *PrintInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 16u> resolved_operands;
    resolved_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        resolved_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return Pool::current()->create<PrintInst>(_format, resolved_operands);
}

}// namespace luisa::compute::xir
