//
// Created by mike on 11/1/25.
//

#include "cuda_codegen_llvm_impl.h"

namespace luisa::compute::cuda {

llvm::Value *CUDACodegenLLVMImpl::_convert_llvm_reg_value_to_mem(IB &b, llvm::Value *reg_v, llvm::Type *mem_type) noexcept {
    auto reg_type = reg_v->getType();
    if (reg_type == mem_type) { return reg_v; }
    // i1 scalars and vectors are stored as i8 in memory
    if (reg_type->isIntOrIntVectorTy(1)) {
        LUISA_DEBUG_ASSERT(mem_type->isIntOrIntVectorTy(8));
        return b.CreateZExt(reg_v, mem_type, reg_v->getName().str() + ".to.mem");
    }
    // otherwise it must be i64/f64 vector types
    LUISA_DEBUG_ASSERT(reg_type->isVectorTy());
    auto elem_type = llvm::cast<llvm::VectorType>(reg_type)->getElementType();
    auto elem_count = llvm::cast<llvm::VectorType>(reg_type)->getElementCount().getFixedValue();
    LUISA_DEBUG_ASSERT(elem_count == 3 || elem_count == 4);
    // i64/f64 vectors are stored as padded arrays in memory
    auto padded_elem_count = elem_count == 3 ? 4 : elem_count;
    LUISA_DEBUG_ASSERT(mem_type->isArrayTy() &&
                       mem_type->getArrayElementType() == elem_type &&
                       mem_type->getArrayNumElements() == padded_elem_count);
    auto mem_v = static_cast<llvm::Value *>(llvm::PoisonValue::get(mem_type));
    for (auto i = 0u; i < elem_count; i++) {
        auto reg_elem = b.CreateExtractElement(reg_v, i);
        mem_v = b.CreateInsertValue(mem_v, reg_elem, i);
    }
    mem_v->setName(reg_v->getName().str() + ".to.mem");
    return mem_v;
}

llvm::Value *CUDACodegenLLVMImpl::_convert_llvm_mem_value_to_reg(IB &b, llvm::Value *mem_v, llvm::Type *reg_type) noexcept {
    auto mem_type = mem_v->getType();
    if (mem_type == reg_type) { return mem_v; }
    // i1 scalars and vectors are stored as i8 in memory
    if (reg_type->isIntOrIntVectorTy(1)) {
        LUISA_DEBUG_ASSERT(mem_type->isIntOrIntVectorTy(8));
        return b.CreateTrunc(mem_v, reg_type, mem_v->getName().str() + ".to.reg");
    }
    // otherwise it must be i64/f64 vector types
    LUISA_DEBUG_ASSERT(reg_type->isVectorTy());
    auto elem_type = llvm::cast<llvm::VectorType>(reg_type)->getElementType();
    auto elem_count = llvm::cast<llvm::VectorType>(reg_type)->getElementCount().getFixedValue();
    LUISA_DEBUG_ASSERT(elem_count == 3 || elem_count == 4);
    // i64/f64 vectors are stored as padded arrays in memory
    auto padded_elem_count = elem_count == 3 ? 4 : elem_count;
    LUISA_DEBUG_ASSERT(mem_type->isArrayTy() &&
                       mem_type->getArrayElementType() == elem_type &&
                       mem_type->getArrayNumElements() == padded_elem_count);
    auto reg_v = static_cast<llvm::Value *>(llvm::PoisonValue::get(reg_type));
    for (auto i = 0u; i < elem_count; i++) {
        auto mem_elem = b.CreateExtractValue(mem_v, i);
        reg_v = b.CreateInsertElement(reg_v, mem_elem, i);
    }
    reg_v->setName(mem_v->getName().str() + ".to.reg");
    return reg_v;
}

llvm::Value *CUDACodegenLLVMImpl::_bitwise_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    auto llvm_src_type = _get_llvm_type(src_type);
    auto llvm_dst_type = _get_llvm_type(dst_type);
    auto llvm_src_mem = _convert_llvm_reg_value_to_mem(b, llvm_src, llvm_src_type->mem_type);
    // scalars or vectors, we can use the built-in llvm bitwise cast
    if ((src_type->is_scalar() || src_type->is_vector()) && (dst_type->is_scalar() || dst_type->is_vector())) {
        auto llvm_dst_mem = b.CreateBitCast(llvm_src_mem, llvm_dst_type->mem_type);
        return _convert_llvm_mem_value_to_reg(b, llvm_dst_mem, llvm_dst_type->reg_type);
    }
    // generic, we make a temporary alloca, store the src value, and load as the dst type
    auto llvm_temp = with_insertion_point_backed_up(b, [&] {
        b.SetInsertPoint(&func_ctx.llvm_alloca_block->front());
        return b.CreateAlloca(llvm_src_type->mem_type);
    });
    b.CreateAlignedStore(llvm_src_mem, llvm_temp, llvm::Align{src_type->alignment()});
    auto llvm_dst_mem = b.CreateAlignedLoad(llvm_dst_type->mem_type, llvm_temp, llvm::Align{dst_type->alignment()});
    return _convert_llvm_mem_value_to_reg(b, llvm_dst_mem, llvm_dst_type->reg_type);
}

llvm::Value *CUDACodegenLLVMImpl::_static_cast(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    // scalar to scalar
    if (src_type->is_scalar() && dst_type->is_scalar()) {
        return _static_cast_scalar_to_scalar(b, func_ctx, llvm_src, src_type, dst_type);
    }
    // scalar to vector
    if (src_type->is_scalar() && dst_type->is_vector()) {
        return _static_cast_scalar_to_vector(b, func_ctx, llvm_src, src_type, dst_type);
    }
    // vector to vector
    if (src_type->is_vector() && dst_type->is_vector()) {
        return _static_cast_vector_to_vector(b, func_ctx, llvm_src, src_type, dst_type);
    }
    // other conversions are not supported
    LUISA_ERROR_WITH_LOCATION("Invalid types for static cast: {} -> {}",
                              src_type->description(), dst_type->description());
}

llvm::Value *CUDACodegenLLVMImpl::_static_cast_scalar_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    auto dst_elem_type = dst_type->element();
    // cast the src to dst element type first
    auto llvm_dst_elem = _static_cast_scalar_to_scalar(b, func_ctx, llvm_src, src_type, dst_elem_type);
    // splat to vector
    auto dim = dst_type->dimension();
    return b.CreateVectorSplat(dim, llvm_dst_elem, llvm_src->getName().str() + ".to.vec");
}

namespace detail {

[[nodiscard]] inline llvm::Value *cast_llvm_value_based_on_elem_types(CUDACodegenLLVMImpl::IB &b,
                                                                      llvm::Type *llvm_dst_type, llvm::Value *llvm_src,
                                                                      const Type *src_elem_type, const Type *dst_elem_type) noexcept {
    // other types to bool, implemented as src != 0
    if (dst_elem_type->is_bool()) {
        auto zero = llvm::Constant::getNullValue(llvm_src->getType());
        return b.CreateICmpNE(llvm_src, zero, llvm_src->getName().str() + ".to.bool");
    }
    // bool to other types, implemented as selection
    if (src_elem_type->is_bool()) {
        auto llvm_zero = llvm::Constant::getNullValue(llvm_dst_type);
        auto llvm_one = dst_elem_type->is_int() ?
                            llvm::ConstantInt::get(llvm_dst_type, 1) :
                            llvm::ConstantFP::get(llvm_dst_type, 1.);
        return b.CreateSelect(llvm_src, llvm_one, llvm_zero, llvm_src->getName().str() + ".from.bool");
    }
    struct ScalarTraits {
        bool is_fp;
        bool is_signed;// only valid if is_fp == false
    };
    auto get_scalar_traits = [](const Type *type) noexcept -> ScalarTraits {
        if (type->is_float()) { return {.is_fp = true, .is_signed = false}; }
        switch (type->tag()) {
            case Type::Tag::INT8: [[fallthrough]];
            case Type::Tag::INT16: [[fallthrough]];
            case Type::Tag::INT32: [[fallthrough]];
            case Type::Tag::INT64:
                return {.is_fp = false, .is_signed = true};
            case Type::Tag::UINT8: [[fallthrough]];
            case Type::Tag::UINT16: [[fallthrough]];
            case Type::Tag::UINT32: [[fallthrough]];
            case Type::Tag::UINT64:
                return {.is_fp = false, .is_signed = false};
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Invalid scalar type: {}.", type->description());
    };
    auto [src_is_fp, src_is_signed] = get_scalar_traits(src_elem_type);
    auto [dst_is_fp, dst_is_signed] = get_scalar_traits(dst_elem_type);
    // float to other types
    if (src_is_fp) {
        return dst_is_fp     ? b.CreateFPCast(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".fp.to.fp") :
               dst_is_signed ? b.CreateFPToSI(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".fp.to.sint") :
                               b.CreateFPToUI(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".fp.to.uint");
    }
    // int to float
    if (dst_is_fp) {
        return src_is_signed ? b.CreateSIToFP(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".sint.to.fp") :
                               b.CreateUIToFP(llvm_src, llvm_dst_type, llvm_src->getName().str() + ".uint.to.fp");
    }
    // int to int, signedness is determined by src type
    return b.CreateIntCast(llvm_src, llvm_dst_type, src_is_signed, llvm_src->getName().str() + ".int.to.int");
}

}// namespace detail

llvm::Value *CUDACodegenLLVMImpl::_static_cast_scalar_to_scalar(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    auto llvm_dst_type = _get_llvm_type(dst_type)->reg_type;
    return detail::cast_llvm_value_based_on_elem_types(b, llvm_dst_type, llvm_src, src_type, dst_type);
}

llvm::Value *CUDACodegenLLVMImpl::_static_cast_vector_to_vector(IB &b, FunctionContext &func_ctx, llvm::Value *llvm_src, const Type *src_type, const Type *dst_type) noexcept {
    if (src_type == dst_type) { return llvm_src; }
    LUISA_DEBUG_ASSERT(src_type->dimension() == dst_type->dimension());
    auto src_elem_type = src_type->element();
    auto dst_elem_type = dst_type->element();
    auto llvm_dst_type = _get_llvm_type(dst_type)->reg_type;
    return detail::cast_llvm_value_based_on_elem_types(b, llvm_dst_type, llvm_src, src_elem_type, dst_elem_type);
}

llvm::Value *CUDACodegenLLVMImpl::_translate_cast_inst(IB &b, FunctionContext &func_ctx, const xir::CastInst *inst) noexcept {
    auto src = inst->value();
    auto src_type = src->type();
    auto dst_type = inst->type();
    auto llvm_src = _get_llvm_value(b, func_ctx, src);
    switch (inst->op()) {
        case xir::CastOp::STATIC_CAST: return _static_cast(b, func_ctx, llvm_src, src_type, dst_type);
        case xir::CastOp::BITWISE_CAST: return _bitwise_cast(b, func_ctx, llvm_src, src_type, dst_type);
    }
    LUISA_ERROR_WITH_LOCATION("Invalid cast instruction.");
}

}// namespace luisa::compute::cuda
