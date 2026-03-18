//
// Created by mike on 3/18/26.
//

#include "hip_codegen_llvm_impl.h"
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Transforms/Utils/CodeExtractor.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/Support/FileOutputBuffer.h>

#include <luisa/core/clock.h>

#undef None

namespace luisa::compute::hip {

HIPCodegenLLVMImpl::FunctionContext::FunctionContext(llvm::Function *f) noexcept
    : llvm_func{f},
      llvm_alloca_block{llvm::BasicBlock::Create(f->getContext(), "alloca", f)},
      llvm_entry_block{llvm::BasicBlock::Create(f->getContext(), "entry", f)} {
    IB b{llvm_alloca_block};
    b.CreateBr(llvm_entry_block);
}

HIPCodegenLLVMImpl::HIPCodegenLLVMImpl(HIPCodegenLLVMConfig config) noexcept
    : _config{std::move(config)} {
    LUISA_ASSERT(_config.block_size[0] > 0u && _config.block_size[1] > 0u && _config.block_size[2] > 0u,
                 "Block size must be constant and greater than zero for now.");
    Clock clk;
    _initialize();
    LUISA_VERBOSE_WITH_LOCATION("HIP LLVM codegen initialized in {} ms.", clk.toc());
}

void HIPCodegenLLVMImpl::_initialize() noexcept {
    static std::once_flag once_flag;
    std::call_once(once_flag, [] {
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();
        LLVMInitializeAMDGPUAsmParser();
    });

    static auto target = [] {
        std::string error;
        if (auto t = llvm::TargetRegistry::lookupTarget(amdgpu_target_triple, error)) {
            return t;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to lookup target '{}': {}", amdgpu_target_triple, error);
    }();

    llvm::TargetOptions options;
    options.NoTrappingFPMath = true;
    if (_config.enable_fast_math) {
        options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    } else {
        options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
    }

    auto opt_level = llvm::CodeGenOptLevel::Default;
    switch (_config.opt_level) {
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::CodeGenOptLevel::None; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::CodeGenOptLevel::Less; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::CodeGenOptLevel::Default; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::CodeGenOptLevel::Aggressive; break;
    }

    auto cpu_name = fmt::format("gfx{}", _config.amdgpu_arch);
    _target_machine = target->createTargetMachine(
        llvm::Triple{amdgpu_target_triple}, llvm::StringRef{cpu_name}, {},
        options, llvm::Reloc::Static, llvm::CodeModel::Small, opt_level);

    _data_layout = std::make_unique<llvm::DataLayout>(_target_machine->createDataLayout());

    _llvm_module = std::make_unique<llvm::Module>("hip_module", _llvm_context);
    _llvm_module->setTargetTriple(llvm::Triple{amdgpu_target_triple});
    _llvm_module->setDataLayout(*_data_layout);

    auto i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto &llvm_ctx = _llvm_context;
    auto module_flags = _llvm_module->getOrInsertNamedMetadata("llvm.module.flags");
    module_flags->addOperand(llvm::MDNode::get(llvm_ctx, {llvm::ConstantAsMetadata::get(llvm::Constant::getIntegerValue(i32_type, llvm::APInt(32, 1))),
                                                          llvm::MDString::get(llvm_ctx, "amdhsa_code_object_version"),
                                                          llvm::ConstantAsMetadata::get(llvm::Constant::getIntegerValue(i32_type, llvm::APInt(32, 600)))}));

    _llvm_buffer_type = _get_llvm_buffer_type();
    _llvm_texture_type = _get_llvm_texture_type();
    _llvm_bindless_array_type = _get_llvm_bindless_array_type();
    _llvm_bindless_array_slot_type = _get_llvm_bindless_array_slot_type();
    _llvm_accel_type = _get_llvm_accel_type();
    _llvm_accel_instance_type = _get_llvm_accel_instance_type();
    _llvm_ray_type = _get_llvm_ray_type();
    _llvm_surface_hit_type = _get_llvm_surface_hit_type();
    _llvm_procedural_hit_type = _get_llvm_procedural_hit_type();
    _llvm_committed_hit_type = _get_llvm_committed_hit_type();
    _llvm_ray_query_type = _get_llvm_ray_query_type();
}

void HIPCodegenLLVMImpl::_dump_module(const std::filesystem::path &path) const noexcept {
    std::error_code ec;
    llvm::raw_fd_ostream out{path.string(), ec};
    if (ec) {
        LUISA_WARNING_WITH_LOCATION("Failed to open file for dumping LLVM module: {}.", ec.message());
    } else {
        _llvm_module->print(out, nullptr);
    }
}

void HIPCodegenLLVMImpl::_run_optimization_passes() noexcept {
    if (_config.enable_fast_math) {
        for (auto &f : *_llvm_module) {
            for (auto &bb : f) {
                for (auto &inst : bb) {
                    if (llvm::isa<llvm::FPMathOperator>(inst)) {
                        if (inst.getOpcode() == llvm::Instruction::FAdd) {
                            auto flags = llvm::FastMathFlags::getFast();
                            flags.setNoInfs(false);
                            inst.setFastMathFlags(flags);
                        } else {
                            inst.setFast(true);
                        }
                    }
                }
            }
        }
    }

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PipelineTuningOptions PTO;
    PTO.LoopInterleaving = true;
#if LLVM_VERSION_MAJOR >= 21
    PTO.LoopInterchange = true;
#endif
    PTO.LoopVectorization = true;
    PTO.SLPVectorization = true;
    PTO.LoopUnrolling = true;
    PTO.MergeFunctions = true;
    llvm::PassBuilder PB{_target_machine, PTO};
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

#if LLVM_VERSION_MAJOR >= 19
    _target_machine->registerPassBuilderCallbacks(PB);
#else
    _target_machine->registerPassBuilderCallbacks(PB, true);
#endif

    auto opt_level = llvm::OptimizationLevel::O2;
    switch (_config.opt_level) {
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::OptimizationLevel::O0; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::OptimizationLevel::O1; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::OptimizationLevel::O2; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::OptimizationLevel::O3; break;
    }
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_level);
    MPM.run(*_llvm_module, MAM);
}

luisa::string HIPCodegenLLVMImpl::_generate_code() const noexcept {
    std::string code;
    llvm::raw_string_ostream os{code};
    _llvm_module->print(os, nullptr);
    return luisa::string{code};
}

luisa::string HIPCodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    Clock clk;
    _initialize();

    for (auto f : xir_module.function_list()) {
        if (auto def = f->definition()) {
            _translate_function(def);
        }
    }

    if (llvm::verifyModule(*_llvm_module, &llvm::errs())) {
        _dump_module("debug_bad_module.ll");
        LUISA_ERROR_WITH_LOCATION("Module verification failed.");
    }

    static auto dump_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    if (dump_ir) {
        _dump_module("hip_kernel_before_opt.ll");
    }

    _run_optimization_passes();

    for (auto &func : *_llvm_module) {
        func.setAttributes(llvm::AttributeList{});
        if (func.getName() == "kernel_main") {
            func.addFnAttr(llvm::Attribute::NoInline);
        }
    }

    static auto print_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_PRINT_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    if (print_ir) {
        _llvm_module->print(llvm::outs(), nullptr);
    }

    LUISA_INFO_WITH_LOCATION("HIP LLVM codegen completed in {} ms.", clk.toc());
    return _generate_code();
}

}// namespace luisa::compute::hip
