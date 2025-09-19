//
// Created by mike on 9/17/25.
//

#include "cuda_codegen_llvm_impl.h"
#include "cuda_codegen_llvm.h"

namespace luisa::compute::cuda {

luisa::string luisa_compute_cuda_codegen_llvm(const xir::Module &xir_module, const CUDACodegenLLVMConfig &config) noexcept {
    CUDACodegenLLVMImpl impl{config};
    return impl.generate(xir_module);
}

}// namespace luisa::compute::cuda
