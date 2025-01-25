#include <luisa/xir/instructions/resource.h>

namespace luisa::compute::xir {

ResourceQueryInst::ResourceQueryInst(const Type *type, ResourceQueryOp op,
                                     luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{type},
      InstructionOpMixin{op} { set_operands(operands); }

ResourceQueryInst *ResourceQueryInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 8u> cloned_operands;
    cloned_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        cloned_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return Pool::current()->create<ResourceQueryInst>(type(), op(), cloned_operands);
}

ResourceReadInst::ResourceReadInst(const Type *type, ResourceReadOp op,
                                   luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{type},
      InstructionOpMixin{op} { set_operands(operands); }

ResourceReadInst *ResourceReadInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 8u> cloned_operands;
    cloned_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        cloned_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return Pool::current()->create<ResourceReadInst>(type(), op(), cloned_operands);
}

ResourceWriteInst::ResourceWriteInst(ResourceWriteOp op,
                                     luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{nullptr},
      InstructionOpMixin{op} { set_operands(operands); }

ResourceWriteInst *ResourceWriteInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    luisa::fixed_vector<Value *, 8u> cloned_operands;
    cloned_operands.reserve(operand_count());
    for (auto op_use : operand_uses()) {
        cloned_operands.emplace_back(resolver.resolve(op_use->value()));
    }
    return Pool::current()->create<ResourceWriteInst>(op(), cloned_operands);
}

}// namespace luisa::compute::xir
