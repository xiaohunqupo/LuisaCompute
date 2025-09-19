//
// Created by mike on 9/19/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

CUDACodegenLLVMImpl::CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept
    : _config{std::move(config)} {

    // parse libdevice bitcode
    _libdevice_module = [&] {
        llvm::SMDiagnostic error;
        if (auto m = llvm::parseIR(
                llvm::MemoryBufferRef{llvm::StringRef{reinterpret_cast<const char *>(config.libdevice_bitcode.data()),
                                                      config.libdevice_bitcode.size()},
                                      "libdevice.10.bc"},
                error, _llvm_context)) {
            return m;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to parse libdevice bitcode: {}", error.getMessage());
    }();
}

luisa::string CUDACodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    return {};
}

}// namespace luisa::compute::cuda
