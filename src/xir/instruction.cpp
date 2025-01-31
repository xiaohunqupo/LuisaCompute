#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

Instruction::Instruction(const Type *type) noexcept : Super{type} {}

void Instruction::_set_parent_block(BasicBlock *block) noexcept {
    _parent_block = block;
}

bool Instruction::_should_add_self_to_operand_use_lists() const noexcept {
    return is_linked();
}

void Instruction::_remove_self_from_operand_use_lists() noexcept {
    for (auto use : operand_uses()) {
        use->remove_self();
    }
}

void Instruction::_add_self_to_operand_use_lists() noexcept {
    for (auto use : operand_uses()) {
        LUISA_DEBUG_ASSERT(!use->is_linked(), "Use already linked.");
        if (auto value = use->value()) {
            value->use_list().insert_front(use);
        }
    }
}

void Instruction::remove_self() noexcept {
    Super::remove_self();
    _remove_self_from_operand_use_lists();
    _set_parent_block(nullptr);
}

void Instruction::insert_before_self(Instruction *node) noexcept {
    Super::insert_before_self(node);
    node->_add_self_to_operand_use_lists();
    node->_set_parent_block(_parent_block);
}

void Instruction::insert_after_self(Instruction *node) noexcept {
    Super::insert_after_self(node);
    node->_add_self_to_operand_use_lists();
    node->_set_parent_block(_parent_block);
}

void Instruction::replace_self_with(Instruction *node) noexcept {
    replace_all_uses_with(node);
    insert_before_self(node);
    remove_self();
}

const ControlFlowMerge *Instruction::control_flow_merge() const noexcept {
    return const_cast<Instruction *>(this)->control_flow_merge();
}

DerivedInstructionTag SentinelInst::derived_instruction_tag() const noexcept {
    LUISA_ERROR_WITH_LOCATION("Calling SentinelInst::derived_instruction_tag()");
}

Instruction *SentinelInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    LUISA_ERROR_WITH_LOCATION("Calling SentinelInst::clone()");
}

TerminatorInstruction::TerminatorInstruction() noexcept
    : Instruction{nullptr} {}

BranchTerminatorInstruction::BranchTerminatorInstruction() noexcept {
    auto operands = std::array{static_cast<Value *>(nullptr)};
    set_operands(operands);
}

void BranchTerminatorInstruction::set_target_block(BasicBlock *target) noexcept {
    set_operand(operand_index_target, target);
}

BasicBlock *BranchTerminatorInstruction::create_target_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(target_block() == nullptr || overwrite_existing,
                 "Target block already exists.");
    auto new_block = Pool::current()->create<BasicBlock>();
    set_target_block(new_block);
    return new_block;
}

BasicBlock *BranchTerminatorInstruction::target_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_target));
}

const BasicBlock *BranchTerminatorInstruction::target_block() const noexcept {
    return const_cast<BranchTerminatorInstruction *>(this)->target_block();
}

ConditionalBranchTerminatorInstruction::ConditionalBranchTerminatorInstruction(Value *condition) noexcept {
    auto operands = std::array{condition, static_cast<Value *>(nullptr), static_cast<Value *>(nullptr)};
    LUISA_DEBUG_ASSERT(operands[operand_index_condition] == condition, "Condition operand mismatch.");
    set_operands(operands);
}

void ConditionalBranchTerminatorInstruction::set_condition(Value *condition) noexcept {
    set_operand(operand_index_condition, condition);
}

void ConditionalBranchTerminatorInstruction::set_true_target(BasicBlock *target) noexcept {
    set_operand(operand_index_true_target, target);
}

void ConditionalBranchTerminatorInstruction::set_false_target(BasicBlock *target) noexcept {
    set_operand(operand_index_false_target, target);
}

BasicBlock *ConditionalBranchTerminatorInstruction::create_true_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(true_block() == nullptr || overwrite_existing,
                 "True block already exists.");
    auto new_block = Pool::current()->create<BasicBlock>();
    set_true_target(new_block);
    return new_block;
}

BasicBlock *ConditionalBranchTerminatorInstruction::create_false_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(false_block() == nullptr || overwrite_existing,
                 "False block already exists.");
    auto new_block = Pool::current()->create<BasicBlock>();
    set_false_target(new_block);
    return new_block;
}

Value *ConditionalBranchTerminatorInstruction::condition() noexcept {
    return operand(operand_index_condition);
}

const Value *ConditionalBranchTerminatorInstruction::condition() const noexcept {
    return operand(operand_index_condition);
}

BasicBlock *ConditionalBranchTerminatorInstruction::true_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_true_target));
}

const BasicBlock *ConditionalBranchTerminatorInstruction::true_block() const noexcept {
    return const_cast<ConditionalBranchTerminatorInstruction *>(this)->true_block();
}

BasicBlock *ConditionalBranchTerminatorInstruction::false_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_false_target));
}

const BasicBlock *ConditionalBranchTerminatorInstruction::false_block() const noexcept {
    return const_cast<ConditionalBranchTerminatorInstruction *>(this)->false_block();
}

BasicBlock *ControlFlowMerge::create_merge_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(merge_block() == nullptr || overwrite_existing,
                 "Merge block already exists.");
    auto block = Pool::current()->create<BasicBlock>();
    set_merge_block(block);
    return block;
}

}// namespace luisa::compute::xir
