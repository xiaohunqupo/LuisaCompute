//
// Created by mike on 9/27/25.
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
    LUISA_DEBUG_ASSERT(elem_count == 2 || elem_count == 3 || elem_count == 4);
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
    LUISA_DEBUG_ASSERT(elem_count == 2 || elem_count == 3 || elem_count == 4);
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

}// namespace luisa::compute::cuda
