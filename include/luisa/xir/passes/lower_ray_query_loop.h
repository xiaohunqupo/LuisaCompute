#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/unordered_map.h>

namespace luisa::compute::xir {

class Module;
class Function;

class RayQueryLoopInst;
class RayQueryPipelineInst;

struct RayQueryLoopLowerInfo {
    luisa::unordered_map<RayQueryLoopInst *, RayQueryPipelineInst *> lowered_loops;
};

[[nodiscard]] LC_XIR_API RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_function(Function *function) noexcept;
[[nodiscard]] LC_XIR_API RayQueryLoopLowerInfo lower_ray_query_loop_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir

