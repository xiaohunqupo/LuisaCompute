#pragma once

#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/outline.h>

namespace luisa::compute::xir {

// This pass will outline all outline instructions in the module.
// Information about the outlined functions will be returned.

struct OutlineInfo {
    size_t outlined_func_count{0u};
};

LUISA_XIR_API OutlineInfo outline_pass_run_on_function(Module *module, Function *function) noexcept;
LUISA_XIR_API OutlineInfo outline_pass_run_on_module(Module *module) noexcept;

}// namespace luisa::compute::xir
