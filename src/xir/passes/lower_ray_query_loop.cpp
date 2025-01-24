#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>

namespace luisa::compute::xir {

namespace detail {

struct RayQueryLoopSubgraph {
    Value *query_object;
    luisa::unordered_set<BasicBlock *> unordered;
    luisa::vector<BasicBlock *> reverse_post_order;
};

static void collect_ray_query_loop_basic_blocks_post_order(BasicBlock *block, const BasicBlock *merge,
                                                           RayQueryLoopSubgraph &subgraph) noexcept {
    if (block != merge && subgraph.unordered.emplace(block).second) {
        block->traverse_successors(true, [&](BasicBlock *succ) noexcept {
            collect_ray_query_loop_basic_blocks_post_order(succ, merge, subgraph);
        });
        // note that we are collecting post-order here
        subgraph.reverse_post_order.emplace_back(block);
    }
}

[[nodiscard]] static auto collect_ray_query_loop_subgraph(RayQueryLoopInst *loop) noexcept {
    // get dispatch and merge blocks
    auto dispatch_block = loop->dispatch_block();
    LUISA_DEBUG_ASSERT(dispatch_block != nullptr, "Invalid ray query loop dispatch block.");
    // get query object from dispatch block
    auto dispatch_inst = dispatch_block->terminator();
    LUISA_DEBUG_ASSERT(dispatch_inst != nullptr && dispatch_inst == &dispatch_block->instructions().front() &&
                           dispatch_inst->derived_instruction_tag() == DerivedInstructionTag::RAY_QUERY_DISPATCH,
                       "Invalid ray query loop dispatch instruction.");
    auto query_object = static_cast<RayQueryDispatchInst *>(dispatch_inst)->query_object();
    LUISA_DEBUG_ASSERT(query_object != nullptr, "Invalid ray query loop query object.");
    auto loop_merge = loop->control_flow_merge();
    LUISA_DEBUG_ASSERT(loop_merge != nullptr, "Invalid ray query loop control flow merge.");
    auto merge_block = loop_merge->merge_block();
    LUISA_DEBUG_ASSERT(merge_block != nullptr, "Invalid ray query loop merge block.");
    // collect subgraph
    RayQueryLoopSubgraph subgraph{.query_object = query_object};
    collect_ray_query_loop_basic_blocks_post_order(dispatch_block, merge_block, subgraph);
    // post-order to reverse post-order
    std::reverse(subgraph.reverse_post_order.begin(), subgraph.reverse_post_order.end());
    LUISA_DEBUG_ASSERT(subgraph.reverse_post_order.front() == dispatch_block, "Invalid ray query loop dispatch block.");
    return subgraph;
}

struct RayQueryLoopCaptureList {
    // values that are defined outside the loop but used inside (including
    // variables, excluding the query object and other non-instruction values)
    std::vector<Value *> in_values;
    // values that are defined inside the loop but used outside, which we
    // must create variables for passing them out of the loop
    std::vector<Instruction *> out_values;
};

static void collect_ray_query_loop_capture_list_in_inst(Instruction *inst, const Value *query_object,
                                                        const luisa::unordered_set<Value *> &internal,
                                                        luisa::unordered_set<Value *> &known_in,
                                                        RayQueryLoopCaptureList &list) noexcept {
    // check if any user of the value is outside the loop
    for (auto &&use : inst->use_list()) {
        if (auto user = use.user(); user != nullptr && !internal.contains(user)) {
            list.out_values.emplace_back(inst);
            break;
        }
    }
    // check if any operand of the value is outside the loop
    auto is_interested_value = [&](Value *value) noexcept {
        // check non-null and not query object
        if (value == nullptr || value == query_object) { return false; }
        // check value type
        switch (value->derived_value_tag()) {
            case DerivedValueTag::FUNCTION: [[fallthrough]];
            case DerivedValueTag::BASIC_BLOCK: [[fallthrough]];
            case DerivedValueTag::CONSTANT: [[fallthrough]];
            case DerivedValueTag::SPECIAL_REGISTER: return false;
            case DerivedValueTag::INSTRUCTION: break;
            case DerivedValueTag::ARGUMENT: break;
            default: LUISA_ERROR_WITH_LOCATION("Unknown derived value tag.");
        }
        // check if the value is defined inside the loop
        if (internal.contains(value)) { return false; }
        // check if the value is already known
        return known_in.emplace(value).second;
    };
    for (auto &&op_use : inst->operand_uses()) {
        if (auto op = op_use->value(); is_interested_value(op)) {
            list.in_values.emplace_back(op);
        }
    }
}

[[nodiscard]] static auto collect_ray_query_loop_capture_list(const RayQueryLoopSubgraph &subgraph) noexcept {
    RayQueryLoopCaptureList capture_list;
    luisa::unordered_set<Value *> known_in;
    luisa::unordered_set<Value *> internal;
    for (auto block : subgraph.reverse_post_order) {
        for (auto &&inst : block->instructions()) {
            internal.emplace(&inst);
        }
    }
    for (auto block : subgraph.reverse_post_order) {
        for (auto &&inst : block->instructions()) {
            collect_ray_query_loop_capture_list_in_inst(
                &inst, subgraph.query_object,
                internal,
                known_in, capture_list);
        }
    }
    return capture_list;
}

static void lower_ray_query_loop(Module *module, RayQueryLoopInst *loop, RayQueryLoopLowerInfo &info) noexcept {
    auto subgraph = collect_ray_query_loop_subgraph(loop);
    auto capture_list = collect_ray_query_loop_capture_list(subgraph);
}

static void run_lower_ray_query_loop_pass_on_function(Function *function, RayQueryLoopLowerInfo &info) noexcept {
    if (auto def = function->definition()) {
        luisa::vector<RayQueryLoopInst *> loops;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::RAY_QUERY_LOOP) {
                loops.emplace_back(static_cast<RayQueryLoopInst *>(inst));
            }
        });
        for (auto loop : loops) {
            lower_ray_query_loop(function->module(), loop, info);
        }
    }
}

}// namespace detail

RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_function(Function *function) noexcept {
    RayQueryLoopLowerInfo info;
    detail::run_lower_ray_query_loop_pass_on_function(function, info);
    return info;
}

RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_module(Module *module) noexcept {
    RayQueryLoopLowerInfo info;
    for (auto &&f : module->functions()) {
        detail::run_lower_ray_query_loop_pass_on_function(&f, info);
    }
    return info;
}

}// namespace luisa::compute::xir
