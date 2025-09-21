//
// Created by mike on 9/19/25.
//

#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Target/TargetMachine.h>

#include <luisa/core/logging.h>
#include <luisa/xir/module.h>

#include "cuda_codegen_llvm_config.h"

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig;

class CUDACodegenLLVMImpl {

public:
    static constexpr auto nvptx_target_triple = "nvptx64-nvidia-cuda";

private:
    CUDACodegenLLVMConfig _config;
    llvm::TargetMachine *_target_machine{nullptr};
    llvm::DataLayout _data_layout;
    llvm::LLVMContext _llvm_context;
    std::unique_ptr<llvm::Module> _llvm_module;

private:
    [[nodiscard]] static const llvm::Target *_get_nvptx_target() noexcept;
    void _initialize() noexcept;
    void _run_optimization_passes() noexcept;
    void _dump_module(const std::filesystem::path &path) const noexcept;
    [[nodiscard]] luisa::string _generate_ptx() const noexcept;

public:
    explicit CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept;
    [[nodiscard]] luisa::string generate(const xir::Module &xir_module) noexcept;
};

}// namespace luisa::compute::cuda
