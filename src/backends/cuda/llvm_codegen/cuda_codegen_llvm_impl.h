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
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Target/TargetMachine.h>

#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/xir/module.h>
#include <luisa/xir/builder.h>

#include "cuda_codegen_llvm_config.h"

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig;

class CUDACodegenLLVMImpl {

public:
    static constexpr auto nvptx_target_triple = "nvptx64-nvidia-cuda";
    static constexpr auto nvptx_address_space_global = 1u;
    static constexpr auto nvptx_address_space_shared = 3u;
    static constexpr auto nvptx_address_space_constant = 4u;
    static constexpr auto nvptx_address_space_local = 5u;

    // This struct holds information about the LLVM types mapped from XIR types
    // mem_type is the type that has the same size and smaller or equal alignment as the XIR type for correct memory layout
    // reg_type is the type used in registers, which may be different from mem_type for i1/i64/f64 vector types and aggregate types that contain vector types
    struct LLVMTypeInfo {
        llvm::Type *mem_type;                // The LLVM type used in memory (with proper alignment)
        llvm::Type *reg_type;                // The LLVM type used in registers
        luisa::vector<size_t> member_indices;// For struct type, the mapping from member index to LLVM struct field index
        luisa::vector<size_t> member_offsets;// For struct type, the byte offset of each member
    };

    using IB = llvm::IRBuilder<>;

    struct FunctionContext {
        llvm::Function *llvm_func;
        llvm::BasicBlock *llvm_alloca_block;
        llvm::BasicBlock *llvm_entry_block;
        llvm::Value *llvm_dispatch_size{nullptr};
        llvm::Value *llvm_kernel_id{nullptr};
        llvm::DenseMap<const xir::Value *, llvm::Value *> local_values;
        std::vector<const xir::PhiInst *> pending_phi_nodes;

        explicit FunctionContext(llvm::Function *f) noexcept;

        template<typename T = llvm::Value>
        [[nodiscard]] T *get_local_value(const xir::Value *v) const noexcept {
            LUISA_DEBUG_ASSERT(v != nullptr, "Value is null.");
            auto iter = local_values.find(v);
            LUISA_DEBUG_ASSERT(iter != local_values.end() && llvm::isa<T>(iter->second), "Local value not found.");
            return llvm::cast<T>(iter->second);
        }
    };

private:
    CUDACodegenLLVMConfig _config;
    llvm::TargetMachine *_target_machine{nullptr};
    std::unique_ptr<llvm::DataLayout> _data_layout;
    llvm::LLVMContext _llvm_context;
    std::unique_ptr<llvm::Module> _llvm_module;

    llvm::Type *_llvm_buffer_type{nullptr};             // { ptr, i32 offset, i32 size } as defined in cuda_buffer.h
    llvm::Type *_llvm_texture_type{nullptr};            // { i64 handle, i64 storage } as defined in cuda_texture.h
    llvm::Type *_llvm_bindless_array_type{nullptr};     // { ptr slots, i64 capacity } as defined in cuda_bindless_array.h
    llvm::Type *_llvm_bindless_array_slot_type{nullptr};// { i64 buffer, i64 size, i64 tex2d, i64 tex3d } as defined in cuda_bindless_array.h
    llvm::Type *_llvm_accel_type{nullptr};              // { i64 handle, ptr instances } as defined in cuda_accel.h
    llvm::Type *_llvm_accel_instance_type{nullptr};     // { [ 12 x float ] affine, i32 user_id, i32 sbt_offset, i32 mask, i32 flags, i64 handle, i32 padding } as defined in optix_api.h
    llvm::DenseMap<const Type *, luisa::unique_ptr<LLVMTypeInfo>> _xir_to_llvm_type;

    llvm::DenseMap<const xir::Function *, llvm::Function *> _xir_to_llvm_function;
    llvm::DenseMap<const xir::Constant *, llvm::Constant *> _xir_to_llvm_global;

private:
    [[nodiscard]] static const llvm::Target *_get_nvptx_target() noexcept;
    void _initialize() noexcept;
    void _run_optimization_passes() noexcept;
    void _dump_module(const std::filesystem::path &path) const noexcept;
    [[nodiscard]] luisa::string _generate_ptx() const noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_type.cpp */
    [[nodiscard]] const LLVMTypeInfo *_get_llvm_type(const Type *type) noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_buffer_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_texture_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_bindless_array_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_bindless_array_slot_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_accel_type() noexcept;
    [[nodiscard]] llvm::Type *_get_llvm_accel_instance_type() noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_func.cpp */
    [[nodiscard]] llvm::Function *_get_llvm_function(const xir::Function *func) noexcept;
    [[nodiscard]] llvm::Function *_create_llvm_kernel_function(const xir::KernelFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_create_llvm_callable_function(const xir::CallableFunction *func) noexcept;
    [[nodiscard]] llvm::Function *_create_llvm_external_function(const xir::ExternalFunction *func) noexcept;
    [[nodiscard]] llvm::BasicBlock *_translate_function_definition(FunctionContext &func_ctx, const xir::FunctionDefinition *f) noexcept;
    static void _mark_llvm_function_as_pure(llvm::Function *func) noexcept;
    [[nodiscard]] llvm::Function *_get_assert_function() noexcept;
    [[nodiscard]] llvm::Function *_get_vprintf_function() noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_const.cpp */
    [[nodiscard]] llvm::Value *_get_llvm_literal(IB &b, const Type *type, const void *data) noexcept;
    [[nodiscard]] llvm::Value *_get_llvm_constant(IB &b, const xir::Constant *c) noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_sreg.cpp */
    [[nodiscard]] llvm::Value *_read_special_register(IB &b, const FunctionContext &func_ctx,
                                                      xir::DerivedSpecialRegisterTag tag) noexcept;
    [[nodiscard]] llvm::Value *_read_block_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_block_size(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_thread_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_dispatch_size(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_dispatch_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_warp_size(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_warp_lane_id(IB &b, const FunctionContext &func_ctx) noexcept;
    [[nodiscard]] llvm::Value *_read_kernel_id(IB &b, const FunctionContext &func_ctx) noexcept;

    /* the following methods are defined in cuda_codegen_llvm_impl_inst.cpp */
    [[nodiscard]] static llvm::Value *_create_llvm_vector(IB &b, llvm::ArrayRef<llvm::Value *> elems) noexcept;
    [[nodiscard]] static llvm::Value *_convert_llvm_reg_value_to_mem(IB &b, llvm::Value *reg_v, llvm::Type *mem_type) noexcept;
    [[nodiscard]] static llvm::Value *_convert_llvm_mem_value_to_reg(IB &b, llvm::Value *mem_v, llvm::Type *reg_type) noexcept;
    void _translate_instruction(IB &b, FunctionContext &func_ctx, const xir::Instruction *inst) noexcept;
    // control flow instructions: if, switch, loop, simple_loop, branch, conditional_branch, unreachable, break, continue, return, raster_discard
    void _translate_if_inst(IB &b, FunctionContext &func_ctx, const xir::IfInst *inst) noexcept;
    void _translate_switch_inst(IB &b, FunctionContext &func_ctx, const xir::SwitchInst *inst) noexcept;
    void _translate_loop_inst(IB &b, FunctionContext &func_ctx, const xir::LoopInst *inst) noexcept;
    void _translate_simple_loop_inst(IB &b, FunctionContext &func_ctx, const xir::SimpleLoopInst *inst) noexcept;
    void _translate_branch_inst(IB &b, FunctionContext &func_ctx, const xir::BranchInst *inst) noexcept;
    void _translate_conditional_branch_inst(IB &b, FunctionContext &func_ctx, const xir::ConditionalBranchInst *inst) noexcept;
    void _translate_unreachable_inst(IB &b, FunctionContext &func_ctx, const xir::UnreachableInst *inst) noexcept;
    void _translate_break_inst(IB &b, FunctionContext &func_ctx, const xir::BreakInst *inst) noexcept;
    void _translate_continue_inst(IB &b, FunctionContext &func_ctx, const xir::ContinueInst *inst) noexcept;
    void _translate_return_inst(IB &b, FunctionContext &func_ctx, const xir::ReturnInst *inst) noexcept;
    void _translate_raster_discard_inst(IB &b, FunctionContext &func_ctx, const xir::RasterDiscardInst *inst) noexcept;
    // PHI nodes
    [[nodiscard]] llvm::PHINode *_translate_phi_inst(IB &b, FunctionContext &func_ctx, const xir::PhiInst *inst) noexcept;
    void _finalize_pending_phi_nodes(const FunctionContext &func_ctx) noexcept;
    // variable instructions: alloca, load, store, gep
    [[nodiscard]] llvm::Value *_translate_alloca_inst(IB &b, FunctionContext &func_ctx, const xir::AllocaInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_load_inst(IB &b, FunctionContext &func_ctx, const xir::LoadInst *inst) noexcept;
    void _translate_store_inst(IB &b, FunctionContext &func_ctx, const xir::StoreInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_gep_inst(IB &b, FunctionContext &func_ctx, const xir::GEPInst *inst) noexcept;
    // atomic instructions
    [[nodiscard]] llvm::Value *_translate_atomic_inst(IB &b, FunctionContext &func_ctx, const xir::AtomicInst *inst) noexcept;
    // arithmetic instructions
    [[nodiscard]] llvm::Value *_translate_arithmetic_inst(IB &b, FunctionContext &func_ctx, const xir::ArithmeticInst *inst) noexcept;
    // thread group instructions
    [[nodiscard]] llvm::Value *_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept;
    // resource instructions: resource_query, resource_read, resource_write
    [[nodiscard]] llvm::Value *_translate_resource_query_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceQueryInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_resource_read_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceReadInst *inst) noexcept;
    void _translate_resource_write_inst(IB &b, FunctionContext &func_ctx, const xir::ResourceWriteInst *inst) noexcept;
    // ray query instructions: ray_query_loop, ray_query_dispatch, ray_query_object_read, ray_query_object_write, ray_query_pipeline
    void _translate_ray_query_loop_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryLoopInst *inst) noexcept;
    void _translate_ray_query_dispatch_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryDispatchInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_ray_query_object_read_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectReadInst *inst) noexcept;
    void _translate_ray_query_object_write_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryObjectWriteInst *inst) noexcept;
    void _translate_ray_query_pipeline_inst(IB &b, FunctionContext &func_ctx, const xir::RayQueryPipelineInst *inst) noexcept;
    // autodiff instructions: autodiff_scope, autodiff_intrinsic
    void _translate_autodiff_scope_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffScopeInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_autodiff_intrinsic_inst(IB &b, FunctionContext &func_ctx, const xir::AutodiffIntrinsicInst *inst) noexcept;
    // other instructions: call, cast, print, clock, debug_break, assert, assume
    [[nodiscard]] llvm::Value *_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_cast_inst(IB &b, FunctionContext &func_ctx, const xir::CastInst *inst) noexcept;
    void _translate_print_inst(IB &b, FunctionContext &func_ctx, const xir::PrintInst *inst) noexcept;
    [[nodiscard]] llvm::Value *_translate_clock_inst(IB &b, FunctionContext &func_ctx, const xir::ClockInst *inst) noexcept;
    void _translate_debug_break_inst(IB &b, FunctionContext &func_ctx, const xir::DebugBreakInst *inst) noexcept;
    void _translate_assert_inst(IB &b, FunctionContext &func_ctx, const xir::AssertInst *inst) noexcept;
    void _translate_assume_inst(IB &b, FunctionContext &func_ctx, const xir::AssumeInst *inst) noexcept;
    // outline instruction
    void _translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept;

public:
    explicit CUDACodegenLLVMImpl(CUDACodegenLLVMConfig config) noexcept;
    [[nodiscard]] luisa::string generate(const xir::Module &xir_module) noexcept;
};

}// namespace luisa::compute::cuda
