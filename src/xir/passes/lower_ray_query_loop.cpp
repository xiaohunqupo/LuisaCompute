#include <luisa/core/logging.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/dce.h>
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
    luisa::vector<Value *> in_values;
    // values that are defined inside the loop but used outside, which we
    // must create variables for passing them out of the loop
    luisa::vector<Instruction *> out_values;
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

[[nodiscard]] static auto outline_ray_query_loop_dispatch_branch(Module *module, BasicBlock *branch, const BasicBlock *merge,
                                                                 const RayQueryLoopCaptureList &capture_list,
                                                                 luisa::string_view comment) noexcept {
    auto function = module->create_callable(nullptr);
    function->add_comment(comment);
    luisa::unordered_map<Value *, Value *> value_map;
    for (auto in_value : capture_list.in_values) {
        auto in_arg = function->create_argument(in_value->type(), in_value->is_lvalue());
        value_map.emplace(in_value, in_arg);
    }
    // translate the branch block
    auto body = function->create_body_block();
    Builder b;
    b.set_insertion_point(body);
    // TODO: implement translation

    // store out values to out arguments
    for (auto out_value : capture_list.out_values) {
        auto out_arg = function->create_reference_argument(out_value->type());
        if (auto iter = value_map.find(out_value); iter != value_map.end()) {
            b.store(out_arg, iter->second);
        }
    }
    // return
    b.return_void();
    return function;
}

static void lower_ray_query_loop(Function *function, RayQueryLoopInst *loop, RayQueryLoopLowerInfo &info) noexcept {
    auto subgraph = collect_ray_query_loop_subgraph(loop);
    auto capture_list = collect_ray_query_loop_capture_list(subgraph);
    auto dispatch = static_cast<RayQueryDispatchInst *>(subgraph.reverse_post_order.front()->terminator());
    auto merge_block = loop->control_flow_merge()->merge_block();
    LUISA_DEBUG_ASSERT(function->module() != nullptr, "Invalid function module.");
    auto on_surface = outline_ray_query_loop_dispatch_branch(
        function->module(), dispatch->on_surface_candidate_block(),
        merge_block, capture_list, "on_surface function outlined from ray query loop");
    auto on_procedural = outline_ray_query_loop_dispatch_branch(
        function->module(), dispatch->on_procedural_candidate_block(),
        merge_block, capture_list, "on_procedural function outlined from ray query loop");
    // prepare captured arguments
    luisa::vector<Value *> captured_args;
    captured_args.reserve(capture_list.in_values.size() + capture_list.out_values.size());
    for (auto in_value : capture_list.in_values) {
        captured_args.emplace_back(in_value);
    }
    // create variables for out values
    if (!capture_list.out_values.empty()) {
        Builder b;
        b.set_insertion_point(&function->definition()->body_block()->instructions().front());
        for (auto out_value : capture_list.out_values) {
            auto variable = b.alloca_local(out_value->type());
            captured_args.emplace_back(variable);
        }
    }
    // create ray query pipeline
    Builder b;
    b.set_insertion_point(loop->prev());
    auto pipeline = b.ray_query_pipeline(subgraph.query_object, on_surface, on_procedural, captured_args);
    // load the out values and replace the uses
    auto out_variables = luisa::span{captured_args}.subspan(capture_list.in_values.size());
    for (auto i = 0u; i < capture_list.out_values.size(); i++) {
        auto old_out_value = capture_list.out_values[i];
        auto out_variable = out_variables[i];
        auto out_value = b.load(old_out_value->type(), out_variable);
        old_out_value->replace_all_uses_with(out_value);
    }
    // remove the loop and move up instructions from the merge block
    loop->remove_self();
    luisa::vector<Instruction *> merge_instructions;
    for (auto &&inst : merge_block->instructions()) {
        merge_instructions.emplace_back(&inst);
    }
    for (auto inst : merge_instructions) {
        inst->remove_self();
        b.append(inst);
    }
    // record the change
    info.lowered_loops.emplace(loop, pipeline);
}

static void run_lower_ray_query_loop_pass_on_function(Function *function, RayQueryLoopLowerInfo &info) noexcept {
    if (auto def = function->definition()) {
        // discover all ray query loops
        luisa::vector<RayQueryLoopInst *> loops;
        def->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->derived_instruction_tag() == DerivedInstructionTag::RAY_QUERY_LOOP) {
                loops.emplace_back(static_cast<RayQueryLoopInst *>(inst));
            }
        });
        // lower each ray query loop
        for (auto loop : loops) {
            lower_ray_query_loop(function, loop, info);
        }
        // remove dead code after lowering using the DCE pass
        if (!loops.empty()) {
            auto dce_info = dce_pass_run_on_function(function);
            LUISA_VERBOSE("Removed {} dead instruction(s) after lowering ray query loop(s).", dce_info.removed_instructions.size());
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
