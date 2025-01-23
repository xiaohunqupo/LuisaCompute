#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/passes/lower_ray_query_loop.h>

namespace luisa::compute::xir {

namespace detail {

static void lower_ray_query_loop(Module *module, RayQueryLoopInst *loop, RayQueryLoopLowerInfo &info) noexcept {
    // TODO: implement this
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
