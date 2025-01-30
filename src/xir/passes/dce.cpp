#include <luisa/core/logging.h>
#include <luisa/core/stl/optional.h>
#include <luisa/xir/instructions/intrinsic.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/builder.h>

namespace luisa::compute::xir {

namespace detail {

static void eliminate_dead_code_in_function(Function *function, DCEInfo &info) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<Instruction *> dead;
        auto all_users_dead = [&](Instruction *inst) noexcept {
            auto is_live = [&dead](Value *value) noexcept {
                return value != nullptr &&
                       (value->derived_value_tag() != DerivedValueTag::INSTRUCTION ||
                        !dead.contains(static_cast<Instruction *>(value)));
            };
            for (auto &&use : inst->use_list()) {
                if (is_live(use.value())) {
                    return false;// not all users are dead
                }
            }
            return true;// no user or all users are dead
        };
        auto collect_if_dead = [&](Instruction *inst) noexcept {
            if (all_users_dead(inst)) {
                dead.emplace(inst);
            }
        };
        // collect all dead instructions
        for (;;) {
            auto prev_size = dead.size();
            definition->traverse_instructions([&](Instruction *inst) noexcept {
                if (!dead.contains(inst)) {
                    switch (inst->derived_instruction_tag()) {
                        case DerivedInstructionTag::PHI: [[fallthrough]];
                        case DerivedInstructionTag::ALLOCA: [[fallthrough]];
                        case DerivedInstructionTag::LOAD: [[fallthrough]];
                        case DerivedInstructionTag::GEP: [[fallthrough]];
                        case DerivedInstructionTag::ARITHMETIC: [[fallthrough]];
                        case DerivedInstructionTag::CAST: [[fallthrough]];
                        case DerivedInstructionTag::CLOCK: [[fallthrough]];
                        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: [[fallthrough]];
                        case DerivedInstructionTag::RESOURCE_QUERY: [[fallthrough]];
                        case DerivedInstructionTag::RESOURCE_READ: {
                            collect_if_dead(inst);
                            break;
                        }
                        case DerivedInstructionTag::INTRINSIC: {
                            auto intrinsic = static_cast<IntrinsicInst *>(inst);
                            switch (intrinsic->op()) {
                                case IntrinsicOp::NOP: [[fallthrough]];
                                case IntrinsicOp::AUTODIFF_GRADIENT: {
                                    collect_if_dead(inst);
                                    break;
                                }
                                default: break;
                            }
                        }
                        default: break;
                    }
                }
            });
            if (dead.size() == prev_size) { break; }
        }
        // remove dead instructions
        for (auto &&inst : dead) {
            info.removed_instructions.emplace(inst);
            inst->remove_self();
        }
    }
}

// if we find a pointer is only written to and never read, we can remove it
[[nodiscard]] static bool is_pointer_write_only(luisa::unordered_set<Instruction *> &known, Instruction *inst) noexcept {
    if (known.contains(inst)) { return true; }
    for (auto &&use : inst->use_list()) {
        if (auto user = use.user()) {
            if (user->derived_value_tag() != DerivedValueTag::INSTRUCTION) {
                return false;
            }
            switch (auto user_inst = static_cast<Instruction *>(user);
                    user_inst->derived_instruction_tag()) {
                case DerivedInstructionTag::STORE: {
                    // store is good
                    break;
                }
                case DerivedInstructionTag::GEP: {
                    // if the GEP is ever read, we can't remove the pointer
                    if (!is_pointer_write_only(known, user_inst)) { return false; }
                    // otherwise we are safe
                    break;
                }
                default: {
                    // just be conservative and assume the pointer is read
                    return false;
                }
            }
        }
    }
    known.emplace(inst);
    return true;
}

static void collect_inst_and_users_recursive(Instruction *inst, luisa::unordered_set<Instruction *> &collected) noexcept {
    if (collected.emplace(inst).second) {
        for (auto &&use : inst->use_list()) {
            if (auto user = use.user()) {
                LUISA_ASSERT(user->derived_value_tag() == DerivedValueTag::INSTRUCTION,
                             "Only instruction can be user.");
                collect_inst_and_users_recursive(static_cast<Instruction *>(user), collected);
            }
        }
    }
}

static void eliminate_dead_alloca_in_function(Function *function, DCEInfo &info) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<Instruction *> dead;
        luisa::unordered_set<Instruction *> known_write_only;
        definition->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::ALLOCA &&
                !dead.contains(inst) && is_pointer_write_only(known_write_only, inst)) {
                collect_inst_and_users_recursive(inst, dead);
            }
        });
        for (auto &&inst : dead) {
            info.removed_instructions.emplace(inst);
            inst->remove_self();
        }
    }
}

[[nodiscard]] static bool is_block_terminated_by_unreachable(BasicBlock *block) noexcept {
    return block->terminator()->derived_instruction_tag() == DerivedInstructionTag::UNREACHABLE;
}

static void eliminate_instructions_in_unreachable_blocks(const luisa::unordered_set<BasicBlock *> &blocks, DCEInfo &info) noexcept {
    luisa::vector<Instruction *> cache;
    for (auto b : blocks) {
        // replace the terminator with an unreachable instruction if it's not already
        if (!is_block_terminated_by_unreachable(b)) {
            b->terminator()->remove_self();
            xir::Builder builder;
            builder.set_insertion_point(b);
            builder.unreachable_();
        }
        // collect all instructions in the unreachable block
        for (auto &&inst : b->instructions()) {
            cache.emplace_back(&inst);
        }
        // pop the terminator
        cache.pop_back();
        // remove all instructions
        for (auto &&inst : cache) {
            inst->remove_self();
            info.removed_instructions.emplace(inst);
        }
        cache.clear();
    }
}

static void propagate_unreachable_marks_in_function(Function *function, DCEInfo &info) noexcept {
    // run a backward dataflow analysis to propagate unreachable marks:
    // we should mark a block as unreachable if all its successors are marked as unreachable
    if (auto definition = function->definition()) {
        luisa::vector<BasicBlock *> postorder;
        definition->traverse_basic_blocks(BasicBlockTraversalOrder::POST_ORDER, [&](BasicBlock *block) noexcept {
            postorder.emplace_back(block);
        });
        luisa::unordered_set<BasicBlock *> unreachable;
        for (;;) {
            auto prev_reachable_count = unreachable.size();
            for (auto block : postorder) {
                if (!unreachable.contains(block)) {
                    if (is_block_terminated_by_unreachable(block)) {
                        unreachable.emplace(block);
                    } else {
                        auto has_any_successor = false;
                        auto all_successors_unreachable = true;
                        block->traverse_successors(false, [&](BasicBlock *succ) noexcept {
                            has_any_successor = true;
                            if (succ != block && !unreachable.contains(succ) &&
                                !is_block_terminated_by_unreachable(succ)) {
                                all_successors_unreachable = false;
                            }
                        });
                        if (has_any_successor && all_successors_unreachable) {
                            unreachable.emplace(block);
                        }
                    }
                }
            }
            if (unreachable.size() == prev_reachable_count) { break; }
        }
        // eliminate all instructions in unreachable blocks
        eliminate_instructions_in_unreachable_blocks(unreachable, info);
    }
}

[[nodiscard]] static luisa::optional<bool> try_evaluate_static_branch_condition(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond != nullptr, "Branch condition must not be null.");
    if (cond->derived_value_tag() != DerivedValueTag::CONSTANT) { return luisa::nullopt; }
    auto static_cond = static_cast<Constant *>(cond);
    LUISA_DEBUG_ASSERT(static_cond->type()->is_bool(), "Branch condition must be a boolean constant.");
    return static_cond->as<bool>();
}

[[nodiscard]] static luisa::optional<SwitchInst::case_value_type> try_evaluate_static_switch_condition(Value *cond) noexcept {
    LUISA_DEBUG_ASSERT(cond != nullptr, "Switch condition must not be null.");
    if (cond->derived_value_tag() != DerivedValueTag::CONSTANT) { return luisa::nullopt; }
    return [static_cond = static_cast<Constant *>(cond)]() noexcept -> SwitchInst::case_value_type {
        switch (auto t = static_cond->type(); t->tag()) {
            case Type::Tag::BOOL: return static_cond->as<bool>();
            case Type::Tag::INT8: return static_cond->as<int8_t>();
            case Type::Tag::UINT8: return static_cond->as<uint8_t>();
            case Type::Tag::INT16: return static_cond->as<int16_t>();
            case Type::Tag::UINT16: return static_cond->as<uint16_t>();
            case Type::Tag::INT32: return static_cond->as<int32_t>();
            case Type::Tag::UINT32: return static_cond->as<uint32_t>();
            case Type::Tag::INT64: return static_cond->as<int64_t>();
            case Type::Tag::UINT64: return static_cond->as<uint64_t>();
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid switch condition type.");
    }();
}

static void eliminate_unreachable_blocks_in_function(Function *function, DCEInfo &info) noexcept {
    if (auto definition = function->definition()) {
        luisa::unordered_set<BasicBlock *> reachable;
        definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            reachable.emplace(block);
        });
        luisa::unordered_set<BasicBlock *> unreachable;
        for (auto b : reachable) {
            // let's find out instructions users' blocks that are not in the reachable set
            b->traverse_instructions([&](Instruction *inst) noexcept {
                for (auto &&use : inst->use_list()) {
                    if (auto user = use.user()) {
                        if (user->derived_value_tag() == DerivedValueTag::INSTRUCTION) {
                            auto user_inst = static_cast<Instruction *>(user);
                            if (auto user_block = user_inst->parent_block();
                                user_block != nullptr && !reachable.contains(user_block)) {
                                unreachable.emplace(user_block);
                            }
                        }
                    }
                }
            });
            // also check if the terminator is a constant branch
            switch (auto terminator = b->terminator(); terminator->derived_instruction_tag()) {
                case DerivedInstructionTag::IF: [[fallthrough]];
                case DerivedInstructionTag::CONDITIONAL_BRANCH: {
                    auto cond_br_inst = static_cast<ConditionalBranchTerminatorInstruction *>(terminator);
                    if (auto static_cond = try_evaluate_static_branch_condition(cond_br_inst->condition())) {
                        unreachable.emplace(*static_cond ? cond_br_inst->false_block() : cond_br_inst->true_block());
                    }
                    break;
                }
                case DerivedInstructionTag::SWITCH: {
                    auto switch_inst = static_cast<SwitchInst *>(terminator);
                    if (auto static_cond = try_evaluate_static_switch_condition(switch_inst->value())) {
                        auto any_match = false;
                        for (auto i = 0u; i < switch_inst->case_count(); i++) {
                            if (switch_inst->case_value(i) == *static_cond) {
                                any_match = true;
                            } else {
                                LUISA_DEBUG_ASSERT(switch_inst->case_block(i) != nullptr, "Switch case block must not be null.");
                                unreachable.emplace(switch_inst->case_block(i));
                            }
                        }
                        if (any_match) {
                            LUISA_DEBUG_ASSERT(switch_inst->default_block() != nullptr, "Switch default block must not be null.");
                            unreachable.emplace(switch_inst->default_block());
                        }
                    }
                    break;
                }
                default: break;
            }
        }
        // eliminate all instructions in unreachable blocks
        eliminate_instructions_in_unreachable_blocks(unreachable, info);
    }
}

static void fix_phi_nodes_in_function(Function *function) noexcept {
    if (auto definition = function->definition()) {
        luisa::vector<PhiIncoming> valid_incomings;
        definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            for (auto &&inst : block->instructions()) {
                if (inst.derived_instruction_tag() == DerivedInstructionTag::PHI) {
                    auto phi = static_cast<PhiInst *>(&inst);
                    valid_incomings.clear();
                    for (auto i = 0u; i < phi->incoming_count(); i++) {
                        auto incoming = phi->incoming(i);
                        if (incoming.block == nullptr || incoming.value == nullptr) { continue; }
                        if (auto incoming_terminator = incoming.block->terminator()) {
                            auto unreachable_from_incoming = true;
                            for (auto op_use : incoming_terminator->operand_uses()) {
                                if (op_use->value() == incoming.block) {
                                    unreachable_from_incoming = false;
                                    break;
                                }
                            }
                            if (unreachable_from_incoming) { continue; }
                        }
                        valid_incomings.emplace_back(incoming);
                    }
                    if (valid_incomings.size() != phi->incoming_count()) {
                        phi->set_incoming_count(valid_incomings.size());
                        for (auto i = 0u; i < valid_incomings.size(); i++) {
                            if (auto old_incoming = phi->incoming(i), new_incoming = valid_incomings[i];
                                old_incoming.value != new_incoming.value ||
                                old_incoming.block != new_incoming.block) {
                                phi->set_incoming(i, new_incoming.value, new_incoming.block);
                            }
                        }
                    }
                }
            }
        });
    }
}

static void fix_control_flow_merges_in_function(Function *function) noexcept {
    if (auto definition = function->definition()) {
        definition->traverse_basic_blocks([&](BasicBlock *block) noexcept {
            if (auto merge = block->terminator()->control_flow_merge()) {
                if (auto merge_block = merge->merge_block();
                    merge_block != nullptr && is_block_terminated_by_unreachable(merge_block)) {
                    merge->set_merge_block(nullptr);
                }
            }
        });
    }
}

static void run_dce_pass_on_function(Function *function, DCEInfo &info) noexcept {
    propagate_unreachable_marks_in_function(function, info);
    eliminate_unreachable_blocks_in_function(function, info);
    fix_phi_nodes_in_function(function);
    fix_control_flow_merges_in_function(function);
    for (;;) {
        auto prev_count = info.removed_instructions.size();
        eliminate_dead_code_in_function(function, info);
        eliminate_dead_alloca_in_function(function, info);
        // if we didn't remove any instruction, we are done
        if (info.removed_instructions.size() == prev_count) { return; }
    }
}

}// namespace detail

DCEInfo dce_pass_run_on_function(Function *function) noexcept {
    DCEInfo info;
    detail::run_dce_pass_on_function(function, info);
    return info;
}

DCEInfo dce_pass_run_on_module(Module *module) noexcept {
    DCEInfo info;
    for (auto &&f : module->functions()) {
        detail::run_dce_pass_on_function(&f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
