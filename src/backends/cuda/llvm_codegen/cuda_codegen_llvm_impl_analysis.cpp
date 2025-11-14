//
// Created by mike on 11/14/25.
//

#include <luisa/xir/metadata/curve_basis.h>
#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

void CUDACodegenLLVMImpl::_analyze_ray_tracing_usage(const xir::Module *module) noexcept {
    llvm::DenseSet<const xir::Function *> visited;
    for (auto f : module->function_list()) {
        // we only start from kernel functions so that unused functions are not analyzed
        if (f->isa<xir::KernelFunction>()) {
            _analyze_ray_tracing_in_function(f, visited);
        }
    }
}

void CUDACodegenLLVMImpl::_analyze_ray_tracing_in_function(const xir::Function *f, llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (auto def = f->definition(); def != nullptr && visited.insert(f).second) {
        for (auto block : def->basic_blocks()) {
            for (auto inst : block->instructions()) {
                // propagate curve basis info
                if (auto curve_set = inst->find_metadata<xir::CurveBasisMD>()) {
                    _rt_analysis.curve_basis_set.propagate(curve_set->curve_basis_set());
                }
                // look for ray tracing related instructions
                switch (inst->derived_instruction_tag()) {
                    case xir::DerivedInstructionTag::RESOURCE_QUERY: {
                        switch (static_cast<const xir::ResourceQueryInst *>(inst)->op()) {
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: {
                                _rt_analysis.uses_ray_tracing = true;
                                break;
                            }
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: {
                                _rt_analysis.uses_ray_tracing = true;
                                _rt_analysis.uses_ray_query = true;
                                break;
                            }
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
                                _rt_analysis.uses_ray_tracing = true;
                                _rt_analysis.uses_motion_blur = true;
                                break;
                            }
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
                                _rt_analysis.uses_ray_tracing = true;
                                _rt_analysis.uses_ray_query = true;
                                _rt_analysis.uses_motion_blur = true;
                                break;
                            }
                            default: break;
                        }
                        break;
                    }
                    case xir::DerivedInstructionTag::CALL: {
                        auto call = static_cast<const xir::CallInst *>(inst);
                        _analyze_ray_tracing_in_function(call->callee(), visited);
                        break;
                    }
                    default: break;
                }
            }
        }
    }
}

}// namespace luisa::compute::cuda
