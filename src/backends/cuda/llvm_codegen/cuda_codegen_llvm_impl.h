//
// Created by mike on 9/19/25.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>

#include <luisa/core/stl/string.h>
#include <luisa/core/logging.h>
#include <luisa/xir/module.h>

#include "cuda_codegen_llvm_config.h"

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig;

class CUDACodegenLLVMImpl {

private:
    CUDACodegenLLVMConfig _config;
    llvm::LLVMContext _llvm_context;
    std::unique_ptr<llvm::Module> _libdevice_module;
    std::unique_ptr<llvm::Module> _llvm_module;

public:
    explicit CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept;
    [[nodiscard]] luisa::string generate(const xir::Module &xir_module) noexcept;
};

}// namespace luisa::compute::cuda
