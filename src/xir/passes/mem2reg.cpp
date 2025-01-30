#include <luisa/core/logging.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/passes/mem2reg.h>

namespace luisa::compute::xir {

namespace detail {

[[nodiscard]] static bool is_alloca_promotable(AllocaInst *inst) noexcept {
    // check if it's a local variable
    if (inst->space() != AllocSpace::LOCAL) { return false; }
    // check if it's used as reference in other instructions than load/store
    for (auto &&use : inst->use_list()) {
        if (auto user = use.user()) {
            LUISA_DEBUG_ASSERT(user->derived_value_tag() == DerivedValueTag::INSTRUCTION, "Invalid user.");
            switch (auto user_inst = static_cast<Instruction *>(user); user_inst->derived_instruction_tag()) {
                case DerivedInstructionTag::LOAD: break;
                case DerivedInstructionTag::STORE: break;
                default: return false;
            }
        }
    }
    return true;
}

struct AllocaAnalysis {

    const DomTree &dom;
    const luisa::unordered_map<Instruction *, size_t> inst_indices;
    const luisa::unordered_map<BasicBlock *, size_t> block_indices;

    luisa::unordered_map<BasicBlock *, StoreInst *> def_blocks;
    luisa::unordered_map<BasicBlock *, LoadInst *> use_blocks;
    luisa::unordered_set<BasicBlock *> live_in_blocks;

    void analyze(AllocaInst *inst) noexcept {
        def_blocks.clear();
        use_blocks.clear();
        // find def and use blocks
        for (auto &&use : inst->use_list()) {
            if (auto user = use.user()) {
                LUISA_DEBUG_ASSERT(user->derived_value_tag() == DerivedValueTag::INSTRUCTION, "Invalid user.");
                switch (auto user_inst = static_cast<Instruction *>(user); user_inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::LOAD: {
                        auto [_, success] = use_blocks.emplace(user_inst->parent_block(), static_cast<LoadInst *>(user_inst));
                        LUISA_DEBUG_ASSERT(success, "Invalid state.");
                        break;
                    }
                    case DerivedInstructionTag::STORE: {
                        auto [_, success] = def_blocks.emplace(user_inst->parent_block(), static_cast<StoreInst *>(user_inst));
                        LUISA_DEBUG_ASSERT(success, "Invalid state.");
                        break;
                    }
                    default: break;
                }
            }
        }
        // compute live-in blocks
        live_in_blocks.clear();
        luisa::fixed_vector<BasicBlock *, 64u> work_list;
        work_list.reserve(use_blocks.size());
        for (auto [use_block, _] : use_blocks) { work_list.emplace_back(use_block); }
        // extend the live-in block set by adding all non-defining predecessors of the known live-in blocks
        while (!work_list.empty()) {
            auto block = work_list.back();
            work_list.pop_back();
            if (live_in_blocks.emplace(block).second) {
                block->traverse_predecessors(true, [&](BasicBlock *pred) noexcept {
                    if (!def_blocks.contains(pred) && !live_in_blocks.contains(pred)) {
                        work_list.emplace_back(pred);
                    }
                });
            }
        }
    }
};

struct PhiInsertion {

    luisa::unordered_set<BasicBlock *> blocks;

    // the following fields are used across the processing of different alloca's
    luisa::unordered_map<PhiInst *, AllocaInst *> phi_to_alloca;

    void place(AllocaInst *inst, AllocaAnalysis &analysis, Mem2RegInfo &info) noexcept {
        // compute blocks to insert new phi nodes by traversing the closure of dominance frontiers of the def blocks
        blocks.clear();
        luisa::fixed_vector<BasicBlock *, 64u> work_list;
        work_list.reserve(analysis.def_blocks.size());
        for (auto [def_block, _] : analysis.def_blocks) { work_list.emplace_back(def_block); }
        while (!work_list.empty()) {
            auto block = work_list.back();
            work_list.pop_back();
            for (auto frontier : analysis.dom.node(block)->frontiers()) {
                if (auto fb = frontier->block();
                    analysis.live_in_blocks.contains(fb) && blocks.emplace(fb).second) {
                    work_list.emplace_back(fb);
                }
            }
        }
        // insert the phi nodes and replace the load in the same block if any
        auto type = inst->type();
        for (auto block : blocks) {
            Builder b;
            b.set_insertion_point(block->instructions().head_sentinel());
            auto phi = b.phi(type);
            phi_to_alloca.emplace(phi, inst);
            info.inserted_phi_instructions.emplace(phi);
            if (auto iter = analysis.use_blocks.find(block); iter != analysis.use_blocks.end()) {
                // replace the load instruction
                auto load = iter->second;
                load->replace_all_uses_with(phi);
                load->remove_self();
                info.removed_load_instructions.emplace(load);
                // update the analysis because this block no longer uses the alloca
                analysis.use_blocks.erase(iter);
            }
        }
        // each of the remaining use blocks must be dominated by some def block, or it must
        // contain undefined value, which will be handled by the renaming pass later
        work_list.clear();
        for (auto [use_block, load_inst] : analysis.use_blocks) {
            if (auto node = analysis.dom.node_or_null(use_block)) {
                while (node != analysis.dom.root()) {
                    auto parent = node->parent();
                    LUISA_DEBUG_ASSERT(parent != nullptr, "Invalid parent.");
                    if (auto iter = analysis.def_blocks.find(parent->block()); iter != analysis.def_blocks.end()) {
                        auto store = iter->second;
                        load_inst->replace_all_uses_with(store->value());
                        load_inst->remove_self();
                        info.removed_load_instructions.emplace(load_inst);
                        work_list.emplace_back(use_block);// mark for later removal
                        break;
                    }
                    node = parent;
                }
            }
        }
        for (auto use_block : work_list) {
            analysis.use_blocks.erase(use_block);
        }
        if (!analysis.use_blocks.empty()) {
            LUISA_WARNING_WITH_LOCATION("Detected {} load instruction(s) from undefined local variables.",
                                        analysis.use_blocks.size());
        }
    }
};

using AllocaStoreLoadSequence = luisa::unordered_map<BasicBlock *, luisa::fixed_vector<Instruction *, 16u>>;

// after this function, for each block, the must be at most one store and one load instruction for an alloca, and
// the load instruction must precede the store instruction if both exist
static void simplify_single_block_store_load(AllocaInst *inst, AllocaStoreLoadSequence &seq,
                                             const luisa::unordered_map<Instruction *, size_t> &inst_indices,
                                             Mem2RegInfo &info) noexcept {
    // collect load/store instructions concerning the alloca
    seq.clear();
    for (auto &&use : inst->use_list()) {
        if (auto user = use.user()) {
            LUISA_DEBUG_ASSERT(user->derived_value_tag() == DerivedValueTag::INSTRUCTION, "Invalid user.");
            auto user_inst = static_cast<Instruction *>(user);
            if (auto tag = user_inst->derived_instruction_tag();
                tag == DerivedInstructionTag::LOAD || tag == DerivedInstructionTag::STORE) {
                seq[user_inst->parent_block()].emplace_back(user_inst);
            }
        }
    }
    // sort the load/store instructions per block and eliminate them when possible
    for (auto &&[block, instructions] : seq) {
        std::sort(instructions.begin(), instructions.end(), [&](Instruction *lhs, Instruction *rhs) noexcept {
            return inst_indices.at(lhs) < inst_indices.at(rhs);
        });
        // eliminate redundant loads and overwritten stores
        auto last_store = static_cast<StoreInst *>(nullptr);
        for (auto store_or_load : instructions) {
            switch (store_or_load->derived_instruction_tag()) {
                case DerivedInstructionTag::LOAD: {
                    auto load_inst = static_cast<LoadInst *>(store_or_load);
                    if (last_store != nullptr) {// we can forward the last stored value to this load
                        load_inst->replace_all_uses_with(last_store->value());
                        load_inst->remove_self();
                        info.removed_load_instructions.emplace(load_inst);
                    }
                    break;
                }
                case DerivedInstructionTag::STORE: {
                    if (last_store != nullptr) {// we have overwritten the last store so remove it
                        last_store->remove_self();
                        info.removed_store_instructions.emplace(last_store);
                    }
                    // update the last store
                    last_store = static_cast<StoreInst *>(store_or_load);
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Invalid instruction.");
            }
        }
    }
    // if we find the alloca now is stored to only, we can remove it
    auto all_store = true;
    for (auto &&use : inst->use_list()) {
        if (auto user = use.user();
            user != nullptr &&
            static_cast<Instruction *>(user)->derived_instruction_tag() !=
                DerivedInstructionTag::STORE) {
            all_store = false;
            break;
        }
    }
    if (all_store) {
        inst->remove_self();
        info.promoted_alloca_instructions.emplace(inst);
    }
}

static void promote_alloca_instructions_in_function(Function *f, Mem2RegInfo &info) noexcept {
    if (auto def = f->definition()) {
        // collect local alloca instructions that can be promoted
        luisa::vector<AllocaInst *> promotable;
        luisa::unordered_map<Instruction *, size_t> inst_indices;
        luisa::unordered_map<BasicBlock *, size_t> block_indices;
        def->traverse_basic_blocks(BasicBlockTraversalOrder::REVERSE_POST_ORDER, [&](BasicBlock *block) noexcept {
            block_indices.emplace(block, block_indices.size());
            block->traverse_instructions([&](Instruction *inst) noexcept {
                switch (inst->derived_instruction_tag()) {
                    case DerivedInstructionTag::ALLOCA: {
                        if (auto alloca_inst = static_cast<AllocaInst *>(inst); is_alloca_promotable(alloca_inst)) {
                            promotable.emplace_back(alloca_inst);
                        }
                        break;
                    }
                    case DerivedInstructionTag::LOAD: [[fallthrough]];
                    case DerivedInstructionTag::STORE: {
                        inst_indices.emplace(inst, inst_indices.size());
                        break;
                    }
                    default: break;
                }
            });
        });
        // do some simplification first
        if (!promotable.empty()) {
            AllocaStoreLoadSequence seq;
            for (auto inst : promotable) {
                simplify_single_block_store_load(inst, seq, inst_indices, info);
            }
        }
        // erase the alloca instructions that are already removed
        promotable.erase(
            std::remove_if(promotable.begin(), promotable.end(), [&](AllocaInst *inst) noexcept {
                return info.promoted_alloca_instructions.contains(inst);
            }),
            promotable.end());
        // perform the SSA rewrite pass for the remaining alloca instructions
        if (!promotable.empty()) {
            auto dom = compute_dom_tree(def);
            AllocaAnalysis analysis{.dom = dom,
                                    .inst_indices = inst_indices,
                                    .block_indices = block_indices};
            PhiInsertion insertion;
            for (auto inst : promotable) {
                analysis.analyze(inst);
                insertion.place(inst, analysis, info);
            }
        }
    }
}

}// namespace detail

Mem2RegInfo mem2reg_pass_run_on_function(Function *function) noexcept {
    Mem2RegInfo info;
    detail::promote_alloca_instructions_in_function(function, info);
    return info;
}

Mem2RegInfo mem2reg_pass_run_on_module(Module *module) noexcept {
    Mem2RegInfo info;
    for (auto &&f : module->functions()) {
        detail::promote_alloca_instructions_in_function(&f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
