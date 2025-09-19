//
// Created by mike on 9/19/25.
//

#include "cuda_codegen_llvm_impl.h"

#include <llvm/IR/Constants.h>

namespace luisa::compute::cuda {

CUDACodegenLLVMImpl::CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept
    : _config{std::move(config)} {

    // parse libdevice bitcode
    _llvm_module = [&] {
        llvm::SMDiagnostic error;
        llvm::StringRef bc{reinterpret_cast<const char *>(config.libdevice_bitcode.data()), config.libdevice_bitcode.size()};
        if (auto m = llvm::parseIR(
                llvm::MemoryBufferRef{bc, "libdevice.10.bc"},
                error, _llvm_context)) {
            return m;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to parse libdevice bitcode: {}", error.getMessage());
    }();

    // set the target triple
    _llvm_module->setTargetTriple(nvptx_target_triple);

    // internalize all device functions
    for (auto &&f : *_llvm_module) {
        if (f.getName().starts_with("__nv_")) {
            f.setLinkage(llvm::Function::PrivateLinkage);
        }
    }

    auto parse_llvm_constant_string = [](llvm::Value *c) noexcept -> llvm::StringRef {
        if (auto gv = llvm::dyn_cast<llvm::GlobalVariable>(c)) {
            if (auto init = gv->getInitializer()) {
                if (auto ca = llvm::dyn_cast<llvm::ConstantDataArray>(init)) {
                    if (ca->isCString()) {
                        return ca->getAsCString();
                    }
                }
            }
        }
        return {};
    };

    // handle __nvvm_reflect
    if (auto f = _llvm_module->getFunction("__nvvm_reflect")) {
        auto const_one = llvm::ConstantInt::get(llvm::Type::getInt32Ty(_llvm_context), 1);
        auto const_zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(_llvm_context), 0);
        auto const_arch = llvm::ConstantInt::get(llvm::Type::getInt32Ty(_llvm_context), _config.cuda_arch * 10);
        llvm::SmallVector<llvm::Instruction *> reflected;
        for (auto user : f->users()) {
            if (auto call = llvm::dyn_cast<llvm::CallInst>(user)) {
                // try to parse the argument string
                if (auto s = parse_llvm_constant_string(call->getArgOperand(0)); s == "__CUDA_FTZ") {
                    call->replaceAllUsesWith(_config.enable_fast_math ? const_one : const_zero);
                    reflected.emplace_back(call);
                } else if (s == "__CUDA_PREC_SQRT" || s == "__CUDA_PREC_DIV") {
                    call->replaceAllUsesWith(_config.enable_fast_math ? const_zero : const_one);
                    reflected.emplace_back(call);
                } else if (s == "__CUDA_ARCH") {
                    call->replaceAllUsesWith(const_arch);
                    reflected.emplace_back(call);
                }
            }
        }
        for (auto i : reflected) { i->eraseFromParent(); }
        if (f->user_empty()) { f->eraseFromParent(); }
    }

    // dump
    // {
    //     std::error_code ec;
    //     llvm::raw_fd_ostream out{"libdevice.ll", ec};
    //     _llvm_module->print(out, nullptr);
    // }
}

luisa::string CUDACodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    _llvm_module = std::make_unique<llvm::Module>(xir_module.name().value_or("cuda_kernel"), _llvm_context);
    return {};
}

}// namespace luisa::compute::cuda
