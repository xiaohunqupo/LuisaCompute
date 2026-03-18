//
// Created by mike on 3/18/26.
//

#include <luisa/xir/passes/dom_tree.h>
#include <luisa/ast/function.h>
#include <luisa/runtime/rhi/pixel.h>

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Function *HIPCodegenLLVMImpl::_get_or_declare_llvm_function(const xir::Function *func) noexcept {
    if (auto iter = _xir_to_llvm_global.find(func); iter != _xir_to_llvm_global.end()) {
        LUISA_DEBUG_ASSERT(llvm::isa<llvm::Function>(iter->second), "Global is not a function.");
        return static_cast<llvm::Function *>(iter->second);
    }
    auto llvm_func = [this, func]() noexcept {
        switch (func->derived_function_tag()) {
            case xir::DerivedFunctionTag::KERNEL: return _declare_llvm_kernel_function(static_cast<const xir::KernelFunction *>(func));
            case xir::DerivedFunctionTag::CALLABLE: return _declare_llvm_callable_function(static_cast<const xir::CallableFunction *>(func));
            case xir::DerivedFunctionTag::EXTERNAL: return _declare_llvm_external_function(static_cast<const xir::ExternalFunction *>(func));
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported function type.");
    }();
    auto [iter, success] = _xir_to_llvm_global.try_emplace(func, llvm_func);
    LUISA_ASSERT(success, "Failed to insert LLVM function.");
    return llvm_func;
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_kernel_function(const xir::KernelFunction *func) noexcept {
    auto [llvm_func_type, llvm_func_name] = [&]() noexcept -> std::pair<llvm::FunctionType *, llvm::StringRef> {
        auto llvm_void_type = llvm::Type::getVoidTy(_llvm_context);
        auto arg_struct_info = _get_kernel_argument_struct(func);
        return std::make_pair(llvm::FunctionType::get(llvm_void_type, {arg_struct_info->llvm_type}, false), "kernel_main");
    }();
    auto llvm_kernel = llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage, llvm_func_name, _llvm_module.get());
    llvm_kernel->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    return llvm_kernel;
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_callable_function(const xir::CallableFunction *func) noexcept {
    llvm::SmallVector<llvm::Type *> llvm_arg_types;
    for (auto arg : func->arguments()) {
        if (arg->is_reference()) {
            llvm_arg_types.emplace_back(llvm::PointerType::get(_llvm_context, 0));
        } else {
            llvm_arg_types.emplace_back(_get_llvm_type(arg->type())->reg_type);
        }
    }
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_i32x3_type = llvm::FixedVectorType::get(llvm_i32_type, 3);
    llvm_arg_types.emplace_back(llvm_i32x3_type);
    llvm_arg_types.emplace_back(llvm_i32_type);
    auto llvm_ret_type = func->type() == nullptr ? llvm::Type::getVoidTy(_llvm_context) :
                                                   _get_llvm_type(func->type())->reg_type;
    auto llvm_func_type = llvm::FunctionType::get(llvm_ret_type, llvm_arg_types, false);
    auto llvm_func = llvm::Function::Create(llvm_func_type, llvm::Function::PrivateLinkage, 0,
                                            func->name().value_or("callable"), _llvm_module.get());
    return llvm_func;
}

llvm::Function *HIPCodegenLLVMImpl::_declare_llvm_external_function(const xir::ExternalFunction *func) noexcept {
    LUISA_NOT_IMPLEMENTED("External function declaration not implemented.");
}

llvm::Function *HIPCodegenLLVMImpl::_translate_function(const xir::FunctionDefinition *func) noexcept {
    switch (func->derived_function_tag()) {
        case xir::DerivedFunctionTag::KERNEL: return _translate_kernel_function(static_cast<const xir::KernelFunction *>(func));
        case xir::DerivedFunctionTag::CALLABLE: return _translate_callable_function(static_cast<const xir::CallableFunction *>(func));
        case xir::DerivedFunctionTag::EXTERNAL: LUISA_ERROR_WITH_LOCATION("Cannot translate external function.");
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Unsupported function type.");
}

llvm::Function *HIPCodegenLLVMImpl::_translate_kernel_function(const xir::KernelFunction *func) noexcept {
    auto arg_struct_info = _get_kernel_argument_struct(func);
    auto llvm_kernel = _get_or_declare_llvm_function(func);
    LUISA_DEBUG_ASSERT(llvm_kernel->isDeclaration(), "Kernel function already defined.");
    FunctionContext func_ctx{llvm_kernel};
    IB b{func_ctx.llvm_entry_block};
    auto llvm_arg_struct = llvm_kernel->getArg(0);
    auto arg_index = 0u;
    for (auto arg : func->arguments()) {
        auto member_index = arg_struct_info->argument_indices[arg_index];
        auto llvm_member_mem = b.CreateExtractValue(llvm_arg_struct, member_index, arg->name().value_or(""));
        auto llvm_member_reg = arg->is_value() ? _convert_llvm_mem_value_to_reg(b, llvm_member_mem, arg->type()) : llvm_member_mem;
        func_ctx.local_values.try_emplace(arg, llvm_member_reg);
        arg_index++;
    }
    auto llvm_dispatch_size_and_kernel_id = b.CreateExtractValue(llvm_arg_struct, arg_struct_info->dispatch_size_and_kernel_id_index);
    auto llvm_dispatch_size_x = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 0);
    b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_x, b.getInt32(0)));
    auto llvm_dispatch_size_y = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 1);
    b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_y, b.getInt32(0)));
    auto llvm_dispatch_size_z = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 2);
    b.CreateAssumption(b.CreateICmpUGT(llvm_dispatch_size_z, b.getInt32(0)));
    func_ctx.llvm_dispatch_size = _create_llvm_vector(b, {llvm_dispatch_size_x, llvm_dispatch_size_y, llvm_dispatch_size_z});
    func_ctx.llvm_dispatch_size->setName("sreg.dispatch.size");
    func_ctx.llvm_kernel_id = b.CreateExtractValue(llvm_dispatch_size_and_kernel_id, 3, "sreg.kernel.id");
    auto llvm_body = _translate_function_definition(func_ctx, func);
    auto llvm_dispatch_id = _read_dispatch_id(b, func_ctx);
    auto llvm_dispatch_id_in_bounds = b.CreateICmpULT(llvm_dispatch_id, func_ctx.llvm_dispatch_size, "dispatch.id.in.bounds");
    for (int i = 0; i < 3; i++) {
        if (_config.block_size[i] == 1) {
            llvm_dispatch_id_in_bounds = b.CreateInsertElement(llvm_dispatch_id_in_bounds, b.getInt1(true), i);
        }
    }
    auto llvm_dispatch_id_in_bounds_all = b.CreateAndReduce(llvm_dispatch_id_in_bounds);
    auto llvm_exit_block = llvm::BasicBlock::Create(_llvm_context, "exit.early", llvm_kernel);
    b.CreateCondBr(llvm_dispatch_id_in_bounds_all, llvm_body, llvm_exit_block);
    b.SetInsertPoint(llvm_exit_block);
    b.CreateRetVoid();
    return llvm_kernel;
}

llvm::Function *HIPCodegenLLVMImpl::_translate_callable_function(const xir::CallableFunction *func) noexcept {
    auto llvm_func = _get_or_declare_llvm_function(func);
    LUISA_DEBUG_ASSERT(llvm_func->isDeclaration(), "Callable function already defined.");
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
    IB b{func_ctx.llvm_entry_block};
    b.CreateBr(body);
    return llvm_func;
}

namespace {

template<typename F>
void luisa_compute_hip_codegen_llvm_traverse_dom_tree_impl(luisa::unordered_set<const xir::DomTreeNode *> &visited,
                                                           const xir::DomTreeNode *node, const F &f) noexcept {
    if (visited.emplace(node).second) [[likely]] {
        f(node->block());
        for (auto child : node->children()) {
            luisa_compute_hip_codegen_llvm_traverse_dom_tree_impl(visited, child, f);
        }
    }
}

template<typename F>
void luisa_compute_hip_codegen_llvm_traverse_dom_tree(const xir::DomTree &tree, const F &f) noexcept {
    luisa::unordered_set<const xir::DomTreeNode *> visited;
    luisa_compute_hip_codegen_llvm_traverse_dom_tree_impl(visited, tree.root(), f);
}

}// namespace

llvm::BasicBlock *HIPCodegenLLVMImpl::_translate_function_definition(FunctionContext &func_ctx, const xir::FunctionDefinition *f) noexcept {
    for (auto bb : f->basic_blocks()) {
        auto llvm_bb = llvm::BasicBlock::Create(_llvm_context, bb->name().value_or(""), func_ctx.llvm_func);
        func_ctx.local_values.try_emplace(bb, llvm_bb);
    }
    auto dom_tree = xir::compute_dom_tree(const_cast<xir::FunctionDefinition *>(f));
    LUISA_ASSERT(dom_tree.root()->block() == f->body_block());
    luisa_compute_hip_codegen_llvm_traverse_dom_tree(dom_tree, [this, &func_ctx](const xir::BasicBlock *bb) noexcept {
        auto llvm_bb = func_ctx.get_local_value<llvm::BasicBlock>(bb);
        IB b{llvm_bb};
        for (auto inst : bb->instructions()) {
            _translate_instruction(b, func_ctx, inst);
        }
    });
    _finalize_pending_phi_nodes(func_ctx);
    return func_ctx.get_local_value<llvm::BasicBlock>(f->body_block());
}

void HIPCodegenLLVMImpl::_mark_llvm_function_as_pure(llvm::Function *func) noexcept {
    func->addFnAttr(llvm::Attribute::NoCallback);
    func->setMustProgress();
    func->setDoesNotFreeMemory();
    func->setNoSync();
    func->setDoesNotThrow();
    func->setSpeculatable();
    func->setWillReturn();
    func->setDoesNotAccessMemory();
}

llvm::Function *HIPCodegenLLVMImpl::_get_assert_function() noexcept {
    if (auto llvm_f = _llvm_module->getFunction("luisa.assert")) {
        return llvm_f;
    }
    auto llvm_i1_type = llvm::Type::getInt1Ty(_llvm_context);
    auto llvm_const_ptr_type = llvm::PointerType::get(_llvm_context, amdgpu_address_space_constant);
    auto llvm_func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(_llvm_context),
                                                  {llvm_i1_type, llvm_const_ptr_type}, false);
    auto llvm_f = llvm::Function::Create(llvm_func_type,
                                         llvm::Function::PrivateLinkage,
                                         "luisa.assert", *_llvm_module);
    auto llvm_entry = llvm::BasicBlock::Create(_llvm_context, "entry", llvm_f);
    IB b{llvm_entry};
    auto llvm_cond = llvm_f->getArg(0);
    auto llvm_msg = llvm_f->getArg(1);
    llvm_cond->setName("cond");
    llvm_msg->setName("message");
    auto llvm_then_bb = llvm::BasicBlock::Create(_llvm_context, "then", llvm_f);
    auto llvm_trap_bb = llvm::BasicBlock::Create(_llvm_context, "trap", llvm_f);
    b.CreateCondBr(llvm_cond, llvm_then_bb, llvm_trap_bb);
    b.SetInsertPoint(llvm_then_bb);
    b.CreateRetVoid();
    b.SetInsertPoint(llvm_trap_bb);
    auto llvm_vprintf = _get_vprintf_function();
    auto llvm_generic_ptr_type = llvm::PointerType::get(_llvm_context, 0);
    auto llvm_msg_p0 = b.CreateAddrSpaceCast(llvm_msg, llvm_generic_ptr_type);
    auto llvm_null_p0 = llvm::ConstantPointerNull::get(llvm_generic_ptr_type);
    b.CreateCall(llvm_vprintf, {llvm_msg_p0, llvm_null_p0});
    b.CreateIntrinsic(b.getVoidTy(), llvm::Intrinsic::trap, {});
    b.CreateUnreachable();
    return llvm_f;
}

llvm::Function *HIPCodegenLLVMImpl::_get_vprintf_function() noexcept {
    if (auto llvm_f = _llvm_module->getFunction("vprintf")) { return llvm_f; }
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    auto llvm_ptr_type = llvm::PointerType::get(_llvm_context, 0);
    auto llvm_func_type = llvm::FunctionType::get(llvm_i32_type, {llvm_ptr_type, llvm_ptr_type}, false);
    return llvm::Function::Create(llvm_func_type, llvm::Function::ExternalLinkage, "vprintf", *_llvm_module);
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture2d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED("Texture 2D read not implemented for HIP.");
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture2d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED("Texture 2D write not implemented for HIP.");
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture3d_read_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED("Texture 3D read not implemented for HIP.");
}

llvm::Function *HIPCodegenLLVMImpl::_get_texture3d_write_function(llvm::VectorType *llvm_value_type) noexcept {
    LUISA_NOT_IMPLEMENTED("Texture 3D write not implemented for HIP.");
}

llvm::InlineAsm *HIPCodegenLLVMImpl::_get_inline_asm(std::string_view asm_string, std::string_view constraints, bool has_side_effects) noexcept {
    auto map_type = [this](char type) noexcept -> llvm::Type * {
        switch (type) {
            case 'h': return llvm::Type::getInt16Ty(_llvm_context);
            case 'r': return llvm::Type::getInt32Ty(_llvm_context);
            case 'l': return llvm::Type::getInt64Ty(_llvm_context);
            case 'f': return llvm::Type::getFloatTy(_llvm_context);
            case 'd': return llvm::Type::getDoubleTy(_llvm_context);
            default: LUISA_ERROR_WITH_LOCATION("Unsupported inline asm type constraint '{}'.", type);
        }
    };
    llvm::SmallVector<llvm::Type *, 4> param_types;
    llvm::SmallVector<llvm::Type *, 4> return_types;
    auto next_is_output = false;
    for (auto c : constraints) {
        if (c == '=') {
            next_is_output = true;
        } else if (c == ',') {
            next_is_output = false;
        } else {
            auto type = map_type(c);
            if (next_is_output) {
                return_types.emplace_back(type);
            } else {
                param_types.emplace_back(type);
            }
        }
    }
    auto return_type = return_types.empty()     ? llvm::Type::getVoidTy(_llvm_context) :
                       return_types.size() == 1 ? return_types.front() :
                                                  llvm::StructType::get(_llvm_context, return_types);
    auto func_type = llvm::FunctionType::get(return_type, param_types, false);
    return llvm::InlineAsm::get(func_type, asm_string, constraints, has_side_effects);
}

llvm::Value *HIPCodegenLLVMImpl::_translate_call_inst(IB &b, FunctionContext &func_ctx, const xir::CallInst *inst) noexcept {
    auto llvm_callee = _get_or_declare_llvm_function(inst->callee());
    llvm::SmallVector<llvm::Value *> llvm_args;
    llvm_args.reserve(inst->argument_count() + 2u);
    for (auto i = 0u; i < inst->argument_count(); i++) {
        auto llvm_arg = _get_llvm_value(b, func_ctx, inst->argument(i));
        if (auto llvm_arg_type = llvm_arg->getType();
            llvm_arg_type->isPointerTy() && llvm_arg_type->getPointerAddressSpace() != 0) {
            llvm_arg = b.CreateAddrSpaceCast(llvm_arg, b.getPtrTy());
        }
        llvm_args.emplace_back(llvm_arg);
    }
    llvm_args.emplace_back(_read_dispatch_size(b, func_ctx));
    llvm_args.emplace_back(_read_kernel_id(b, func_ctx));
    auto call_inst = b.CreateCall(llvm_callee, llvm_args, inst->name().value_or(""));
    call_inst->setCallingConv(llvm_callee->getCallingConv());
    return inst->type() == nullptr ? nullptr : call_inst;
}

void HIPCodegenLLVMImpl::_translate_outline_inst(IB &b, FunctionContext &func_ctx, const xir::OutlineInst *inst) noexcept {
    LUISA_ERROR_WITH_LOCATION("Outline instruction should have been lowered.");
}

}// namespace luisa::compute::hip
