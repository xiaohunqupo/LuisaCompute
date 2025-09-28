//
// Created by mike on 9/25/25.
//

#include "cuda_codegen_llvm_impl.h"
#include "luisa/xir/passes/dom_tree.h"
#include "luisa/ast/function.h"

namespace luisa::compute::cuda {

llvm::Function *CUDACodegenLLVMImpl::_get_llvm_function(const xir::Function *func) noexcept {
    if (auto iter = _xir_to_llvm_function.find(func); iter != _xir_to_llvm_function.end()) {
        return iter->second;
    }
    auto llvm_func = [this, func]() noexcept {
        switch (func->derived_function_tag()) {
            case xir::DerivedFunctionTag::KERNEL: return _create_llvm_kernel_function(static_cast<const xir::KernelFunction *>(func));
            case xir::DerivedFunctionTag::CALLABLE: return _create_llvm_callable_function(static_cast<const xir::CallableFunction *>(func));
            case xir::DerivedFunctionTag::EXTERNAL: return _create_llvm_external_function(static_cast<const xir::ExternalFunction *>(func));
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported function type.");
    }();
    auto [iter, success] = _xir_to_llvm_function.try_emplace(func, llvm_func);
    LUISA_ASSERT(success, "Failed to insert LLVM function.");
    return iter->second;
}

llvm::Function *CUDACodegenLLVMImpl::_create_llvm_kernel_function(const xir::KernelFunction *func) noexcept {
    llvm::SmallVector<llvm::Type *> llvm_arg_members;
    llvm::SmallVector<llvm::Type *> llvm_arg_reg_types;
    llvm::SmallVector<size_t> llvm_arg_member_indices;
    auto current_offset = static_cast<size_t>(0u);
    static constexpr auto argument_alignment = 16u;
    for (auto arg : func->arguments()) {
        auto next_offset = luisa::align(current_offset, argument_alignment);
        if (next_offset > current_offset) {
            auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
            llvm_arg_members.emplace_back(llvm::ArrayType::get(llvm_i8_type, next_offset - current_offset));
        }
        llvm_arg_member_indices.emplace_back(llvm_arg_members.size());
        auto llvm_arg_type = _get_llvm_type(arg->type());
        llvm_arg_members.emplace_back(llvm_arg_type->mem_type);
        llvm_arg_reg_types.emplace_back(llvm_arg_type->reg_type);
        current_offset = next_offset + _data_layout->getTypeAllocSize(llvm_arg_members.back()).getFixedValue();
    }
    // tail padding and <i32 x 4> for dispatch_size and kernel_id
    if (current_offset % argument_alignment != 0u) {
        auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
        auto count = luisa::align(current_offset, argument_alignment) - current_offset;
        llvm_arg_members.emplace_back(llvm::ArrayType::get(llvm_i8_type, count));
    }
    auto dispatch_size_and_kernel_id_index = llvm_arg_members.size();
    auto llvm_i32x4_type = llvm::VectorType::get(llvm::Type::getInt32Ty(_llvm_context), 4, false);
    llvm_arg_members.emplace_back(llvm_i32x4_type);
    auto llvm_arg_struct_type = llvm::StructType::create(_llvm_context, llvm_arg_members, "kernel.params.struct");
    auto [llvm_kernel, llvm_arg_struct] = [&]() noexcept -> std::pair<llvm::Function *, llvm::Value *> {
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        if (_config.enable_ray_tracing) {// ray tracing kernels use constant memory for args
            auto llvm_global_arg = new ::llvm::GlobalVariable{
                *_llvm_module, llvm_arg_struct_type, true, llvm::GlobalValue::ExternalLinkage,
                nullptr, "params", nullptr, llvm::GlobalValue::NotThreadLocal,
                nvptx_address_space_constant, true};
            llvm_global_arg->setAlignment(llvm::Align{argument_alignment});
            auto llvm_func_type = llvm::FunctionType::get(llvm_void_type, {}, false);
            auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage,
                                                    "__raygen__main", _llvm_module.get());
            return std::make_pair(llvm_func, llvm_global_arg);
        }
        // normal kernels use direct arguments
        auto llvm_func_type = llvm::FunctionType::get(llvm_void_type, {llvm_arg_struct_type}, false);
        auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage,
                                                "kernel_main", _llvm_module.get());
        auto llvm_arg = llvm_func->getArg(0);
        llvm_arg->setName("params");
        return std::make_pair(llvm_func, llvm_arg);
    }();
    llvm_kernel->setCallingConv(llvm::CallingConv::PTX_Kernel);
    FunctionContext func_ctx{llvm_kernel};
    // load arguments
    IB b{func_ctx.llvm_entry_block};
    if (llvm_arg_struct->getType()->isPointerTy()) {
        llvm_arg_struct = b.CreateAlignedLoad(llvm_arg_struct_type, llvm_arg_struct,
                                              llvm::Align{argument_alignment},
                                              "params.load");
    }
    // map arguments to local values
    auto arg_index = 0u;
    for (auto arg : func->arguments()) {
        auto member_index = llvm_arg_member_indices[arg_index];
        auto llvm_member_mem = b.CreateExtractValue(llvm_arg_struct, member_index, arg->name().value_or(""));
        auto llvm_member_reg_type = llvm_arg_reg_types[arg_index];
        auto llvm_member_reg = _convert_llvm_mem_value_to_reg(b, llvm_member_mem, llvm_member_reg_type);
        func_ctx.local_values.try_emplace(arg, llvm_member_reg);
        arg_index++;
    }
    // load dispatch_size_and_kernel_id
    auto llvm_dispatch_size_and_kernel_id = b.CreateExtractValue(llvm_arg_struct, dispatch_size_and_kernel_id_index);
    func_ctx.llvm_dispatch_size = b.CreateShuffleVector(llvm_dispatch_size_and_kernel_id, {0, 1, 2}, "sreg.dispatch.size");
    func_ctx.llvm_kernel_id = b.CreateExtractElement(llvm_dispatch_size_and_kernel_id, 3, "sreg.kernel.id");
    // translate body
    auto llvm_body = _translate_function_definition(func_ctx, func);
    // create guard for out-of-bounds threads if not ray tracing (OptiX will do this for us)
    if (!_config.enable_ray_tracing) {
        auto llvm_dispatch_id = _read_dispatch_id(b, func_ctx);
        auto llvm_dispatch_id_in_bounds = b.CreateICmpULT(llvm_dispatch_id, func_ctx.llvm_dispatch_size, "dispatch.id.in.bounds");
        auto llvm_dispatch_id_in_bounds_all = b.CreateAndReduce(llvm_dispatch_id_in_bounds);
        auto llvm_exit_block = llvm::BasicBlock::Create(_llvm_context, "exit.early", llvm_kernel);
        b.CreateCondBr(llvm_dispatch_id_in_bounds_all, llvm_body, llvm_exit_block);
        // exit block
        b.SetInsertPoint(llvm_exit_block);
        b.CreateRetVoid();
    } else {
        b.CreateBr(llvm_body);
    }
    // branch from entry to body
    return llvm_kernel;
}

llvm::Function *CUDACodegenLLVMImpl::_create_llvm_callable_function(const xir::CallableFunction *func) noexcept {
    llvm::SmallVector<llvm::Type *> llvm_arg_types;
    for (auto arg : func->arguments()) {
        if (arg->is_reference()) {
            llvm_arg_types.emplace_back(llvm::PointerType::get(_llvm_context, 0));
        } else {
            llvm_arg_types.emplace_back(_get_llvm_type(arg->type())->reg_type);
        }
    }
    // the last two arguments are dispatch_size and kernel_id
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i32x3_type = llvm::VectorType::get(llvm_i32_type, 3, false);
    llvm_arg_types.emplace_back(llvm_i32x3_type);// dispatch_size
    llvm_arg_types.emplace_back(llvm_i32_type);  // kernel_id
    auto llvm_ret_type = func->type() == nullptr ? llvm::Type::getVoidTy(_llvm_context) :
                                                   _get_llvm_type(func->type())->reg_type;
    auto llvm_func_type = llvm::FunctionType::get(llvm_ret_type, llvm_arg_types, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, 0,
                                            func->name().value_or("callable"), _llvm_module.get());
    FunctionContext func_ctx{llvm_func};
    auto llvm_arg_iter = llvm_func->arg_begin();
    for (auto arg : func->arguments()) {
        func_ctx.local_values.try_emplace(arg, llvm_arg_iter++);
    }
    func_ctx.llvm_dispatch_size = llvm_arg_iter++;
    func_ctx.llvm_dispatch_size->setName("sreg.dispatch.size");
    func_ctx.llvm_kernel_id = llvm_arg_iter++;
    func_ctx.llvm_kernel_id->setName("sreg.kernel.id");
    auto body = _translate_function_definition(func_ctx, func);
    // branch from entry to body
    IB b{func_ctx.llvm_entry_block};
    b.CreateBr(body);
    return llvm_func;
}

llvm::Function *CUDACodegenLLVMImpl::_create_llvm_external_function(const xir::ExternalFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

namespace {

template<typename F>
void luisa_compute_cuda_codegen_llvm_traverse_dom_tree_impl(luisa::unordered_set<const xir::DomTreeNode *> &visited,
                                                            const xir::DomTreeNode *node, const F &f) noexcept {
    if (visited.emplace(node).second) [[likely]] {
        f(node->block());
        for (auto child : node->children()) {
            luisa_compute_cuda_codegen_llvm_traverse_dom_tree_impl(visited, child, f);
        }
    }
}

template<typename F>
void luisa_compute_cuda_codegen_llvm_traverse_dom_tree(const xir::DomTree &tree, const F &f) noexcept {
    luisa::unordered_set<const xir::DomTreeNode *> visited;
    luisa_compute_cuda_codegen_llvm_traverse_dom_tree_impl(visited, tree.root(), f);
}

}// namespace

llvm::BasicBlock *CUDACodegenLLVMImpl::_translate_function_definition(FunctionContext &func_ctx, const xir::FunctionDefinition *f) noexcept {
    for (auto bb : f->basic_blocks()) {
        auto llvm_bb = llvm::BasicBlock::Create(_llvm_context, bb->name().value_or(""), func_ctx.llvm_func);
        func_ctx.local_values.try_emplace(bb, llvm_bb);
    }
    // generate code for each basic block in dominance order, so that
    // used values are always defined before use except for phi nodes,
    // which are handled separately after all blocks are generated
    auto dom_tree = xir::compute_dom_tree(const_cast<xir::FunctionDefinition *>(f));
    LUISA_ASSERT(dom_tree.root()->block() == f->body_block());
    luisa_compute_cuda_codegen_llvm_traverse_dom_tree(dom_tree, [this, &func_ctx](const xir::BasicBlock *bb) noexcept {
        auto llvm_bb = func_ctx.get_local_value<llvm::BasicBlock>(bb);
        IB b{llvm_bb};
        for (auto inst : bb->instructions()) {
            _translate_instruction(b, func_ctx, inst);
        }
    });
    // finalize phi nodes
    _finalize_pending_phi_nodes(func_ctx);
    return func_ctx.get_local_value<llvm::BasicBlock>(f->body_block());
}

void CUDACodegenLLVMImpl::_mark_llvm_function_as_pure(llvm::Function *func) noexcept {
    func->addFnAttr(llvm::Attribute::NoCallback);
    func->setMustProgress();
    func->setDoesNotFreeMemory();
    func->setNoSync();
    func->setDoesNotThrow();
    func->setSpeculatable();
    func->setWillReturn();
    func->setDoesNotAccessMemory();
}

}// namespace luisa::compute::cuda
