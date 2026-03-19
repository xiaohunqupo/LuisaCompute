//
// Created by mike on 3/19/26.
//

#include "hip_codegen_llvm_impl.h"

namespace luisa::compute::hip {

llvm::Value *HIPCodegenLLVMImpl::_translate_thread_group_inst(IB &b, FunctionContext &func_ctx, const xir::ThreadGroupInst *inst) noexcept {

    // AMDGPU wave32 shuffle via ds_swizzle or readlane/writelane
    auto shuffle_idx = [&](llvm::Value *value, llvm::Value *src_lane) noexcept -> llvm::Value * {
        return b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_readlane, {value, src_lane});
    };

    // butterfly shuffle: xor current lane with offset
    auto shuffle_xor = [&](llvm::Value *value, uint32_t offset) noexcept -> llvm::Value * {
        auto lane_id = _read_warp_lane_id(b, func_ctx);
        auto src_lane = b.CreateXor(lane_id, b.getInt32(offset));
        return shuffle_idx(value, src_lane);
    };

    // shuffle up: read from (lane - offset), clamped
    auto shuffle_up = [&](llvm::Value *value, uint32_t offset) noexcept -> llvm::Value * {
        auto lane_id = _read_warp_lane_id(b, func_ctx);
        auto src_lane = b.CreateSub(lane_id, b.getInt32(offset));
        return shuffle_idx(value, src_lane);
    };

    auto pack_into_i32_vector = [&](llvm::Value *v) noexcept {
        LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy() || v->getType()->isFPOrFPVectorTy());
        auto bitwidth = _data_layout->getTypeSizeInBits(v->getType());
        if (bitwidth < 32) {
            v = b.CreateZExt(b.CreateBitCast(v, b.getIntNTy(bitwidth)), b.getInt32Ty());
        }
        auto n = (bitwidth + 31) / 32;
        return std::make_pair(b.CreateBitCast(v, llvm::VectorType::get(b.getInt32Ty(), n, false)), n);
    };

    auto unpack_from_i32_vector = [&](llvm::Value *v, llvm::Type *target_type) noexcept {
        LUISA_DEBUG_ASSERT(v->getType()->isIntOrIntVectorTy(32));
        LUISA_DEBUG_ASSERT(target_type->isIntOrIntVectorTy() || target_type->isFPOrFPVectorTy());
        auto bitwidth = _data_layout->getTypeSizeInBits(target_type);
        if (bitwidth < 32) {
            v = b.CreateTrunc(b.CreateExtractElement(v, b.getInt32(0)), b.getIntNTy(bitwidth));
        }
        return b.CreateBitCast(v, target_type);
    };

    auto reduce_active = [&](llvm::Value *mask, llvm::Value *lane, llvm::Value *value, auto binary_op) noexcept -> llvm::Value * {
        LUISA_DEBUG_ASSERT(value->getType()->isIntOrIntVectorTy(32));
        auto shuffle = [&](llvm::Value *x, uint32_t offset) noexcept {
            return shuffle_xor(x, offset);
        };
        for (auto offset = 16u; offset >= 1u; offset /= 2u) {
            auto shuffled_value = static_cast<llvm::Value *>(nullptr);
            if (auto vt = llvm::dyn_cast<llvm::VectorType>(value->getType())) {
                llvm::SmallVector<llvm::Value *, 8> shuffled_values;
                auto dim = vt->getElementCount().getFixedValue();
                for (auto i = 0u; i < dim; i++) {
                    auto elem = b.CreateExtractElement(value, i);
                    shuffled_values.emplace_back(shuffle(elem, offset));
                }
                shuffled_value = _create_llvm_vector(b, shuffled_values);
            } else {
                shuffled_value = shuffle(value, offset);
            }
            auto alive_mask = b.CreateShl(b.getInt32(1), b.CreateXor(lane, b.getInt32(offset)));
            auto is_alive = b.CreateICmpNE(b.CreateAnd(mask, alive_mask), b.getInt32(0));
            value = b.CreateSelect(is_alive, binary_op(value, shuffled_value), value);
        }
        return value;
    };

    // prefix scan condition helper
    auto reduce_prefix_cond_func = [this] {
        using namespace std::string_view_literals;
        constexpr auto name = "luisa.warp.prefix.scan.cond"sv;
        auto cond = _llvm_module->getFunction(name);
        if (cond != nullptr) { return cond; }
        auto i1_type = llvm::Type::getInt1Ty(_llvm_module->getContext());
        auto i32_type = llvm::Type::getInt32Ty(_llvm_module->getContext());
        auto func_type = llvm::FunctionType::get(i1_type, {i32_type, i32_type, i32_type}, false);
        cond = llvm::Function::Create(func_type, llvm::Function::PrivateLinkage, name, *_llvm_module);
        cond->addFnAttr(llvm::Attribute::AlwaysInline);
        auto entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", cond);
        IB func_b{entry_bb};
        auto lane = cond->getArg(0);
        auto mask = cond->getArg(1);
        auto offset = cond->getArg(2);
        auto lane_ge_offset = func_b.CreateICmpUGE(lane, offset);
        auto lane_sub_offset = func_b.CreateSub(lane, offset);
        auto lane_shift = func_b.CreateShl(func_b.getInt32(1), lane_sub_offset);
        auto is_alive = func_b.CreateICmpNE(func_b.CreateAnd(mask, lane_shift), func_b.getInt32(0));
        func_b.CreateRet(func_b.CreateAnd(lane_ge_offset, is_alive));
        return cond;
    };

    auto reduce_prefix = [&](llvm::Value *mask, llvm::Value *lane, llvm::Value *unit, llvm::Value *value, auto binary_op) noexcept {
        LUISA_DEBUG_ASSERT(value->getType()->isIntOrIntVectorTy(32));

        // value = shfl_sync_idx(mask, x, active_prev_lane)
        auto prefix_mask = _read_warp_prefix_lane_mask(b, func_ctx);
        auto active_prev_mask = b.CreateAnd(prefix_mask, mask);
        auto active_prev_lane = b.CreateSub(b.getInt32(31),
                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::ctlz, active_prev_mask, b.getInt1(false)),
                                            "", true, true);
        if (auto vt = llvm::dyn_cast<llvm::VectorType>(value->getType())) {
            llvm::SmallVector<llvm::Value *, 8> shuffled_values;
            auto dim = vt->getElementCount().getFixedValue();
            for (auto i = 0u; i < dim; i++) {
                auto elem = b.CreateExtractElement(value, i);
                shuffled_values.emplace_back(shuffle_idx(elem, active_prev_lane));
            }
            value = _create_llvm_vector(b, shuffled_values);
        } else {
            value = shuffle_idx(value, active_prev_lane);
        }
        // value = select(lane == first_active_lane, unit, value)
        auto first_active_lane = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, mask, b.getInt1(true));
        auto is_first_active = b.CreateICmpEQ(lane, first_active_lane);
        value = b.CreateSelect(is_first_active, unit, value);
        // perform shuffles
        auto cond_func = reduce_prefix_cond_func();
        for (auto offset = 1u; offset <= 16u; offset *= 2u) {
            auto shuffled_value = static_cast<llvm::Value *>(nullptr);
            if (auto vt = llvm::dyn_cast<llvm::VectorType>(value->getType())) {
                llvm::SmallVector<llvm::Value *, 8> shuffled_values;
                auto dim = vt->getElementCount().getFixedValue();
                for (auto i = 0u; i < dim; i++) {
                    auto elem = b.CreateExtractElement(value, i);
                    shuffled_values.emplace_back(shuffle_up(elem, offset));
                }
                shuffled_value = _create_llvm_vector(b, shuffled_values);
            } else {
                shuffled_value = shuffle_up(value, offset);
            }
            auto cond = b.CreateCall(cond_func, {lane, mask, b.getInt32(offset)});
            auto new_value = binary_op(value, shuffled_value);
            value = b.CreateSelect(cond, new_value, value);
        }
        return value;
    };

    switch (auto op = inst->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER: {
            // SER is not supported on AMDGPU — no-op
            return nullptr;
        }
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: LUISA_ERROR_WITH_LOCATION("RASTER_QUAD_DDX is not supported in HIP backend.");
        case xir::ThreadGroupOp::RASTER_QUAD_DDY: LUISA_ERROR_WITH_LOCATION("RASTER_QUAD_DDY is not supported in HIP backend.");
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_first_active_lane_id = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            return b.CreateICmpEQ(llvm_lane_id, llvm_first_active_lane_id);
        }
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: {
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            return b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool_or_bool_vector());
            LUISA_DEBUG_ASSERT(inst->operand_count() == 1 && inst->operand(0)->type()->is_scalar_or_vector());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            // shuffle from first active lane and compare
            auto llvm_first_active_lane_id = b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            auto [llvm_packed_value, packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_packed_value_from_first = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_packed_value->getType()));
            for (auto i = 0; i < packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_packed_value, i);
                auto llvm_elem_from_first = shuffle_idx(llvm_local_elem, llvm_first_active_lane_id);
                llvm_packed_value_from_first = b.CreateInsertElement(llvm_packed_value_from_first, llvm_elem_from_first, i);
            }
            auto llvm_value_from_first = unpack_from_i32_vector(llvm_packed_value_from_first, llvm_value_type);
            auto llvm_cmp = llvm_value_type->isFPOrFPVectorTy() ?
                                b.CreateFCmpOEQ(llvm_value, llvm_value_from_first) :
                                b.CreateICmpEQ(llvm_value, llvm_value_from_first);
            // vote all: ballot(cmp) == active_mask
            if (inst->type()->is_bool()) {
                auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ballot, {llvm_cmp});
                return b.CreateICmpEQ(llvm_ballot, llvm_active_mask);
            }
            auto llvm_result = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_cmp->getType()));
            for (auto i = 0u; i < inst->type()->dimension(); i++) {
                auto llvm_elem = b.CreateExtractElement(llvm_cmp, b.getInt32(i));
                auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ballot, {llvm_elem});
                auto llvm_elem_voted = b.CreateICmpEQ(llvm_ballot, llvm_active_mask);
                llvm_result = b.CreateInsertElement(llvm_result, llvm_elem_voted, i);
            }
            return llvm_result;
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: {
            LUISA_DEBUG_ASSERT(inst->operand_count() == 1 && inst->type() == inst->operand(0)->type());
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_value->getType()->isIntOrIntVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            auto [llvm_packed_value, packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_result_packed = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_packed_value->getType()));
            auto handle_one_i32 = [&](llvm::Value *llvm_local_i32) noexcept -> llvm::Value * {
                return reduce_active(llvm_active_mask, llvm_lane_id, llvm_local_i32, [&](auto x, auto y) noexcept {
                    switch (op) {
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: return b.CreateAnd(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: return b.CreateOr(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: return b.CreateXor(x, y);
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid bitwise warp reduction op.");
                });
            };
            for (auto i = 0; i < packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_packed_value, i);
                auto llvm_reduced_elem = handle_one_i32(llvm_local_elem);
                llvm_result_packed = b.CreateInsertElement(llvm_result_packed, llvm_reduced_elem, i);
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value->getType());
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: {
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ballot, {llvm_pred});
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: {
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            LUISA_DEBUG_ASSERT(llvm_value_type->isIntOrIntVectorTy() || llvm_value_type->isFPOrFPVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            auto llvm_packed_value = pack_into_i32_vector(llvm_value).first;
            auto llvm_result_packed = reduce_active(llvm_active_mask, llvm_lane_id, llvm_packed_value, [&](auto x, auto y) noexcept {
                x = unpack_from_i32_vector(x, llvm_value_type);
                y = unpack_from_i32_vector(y, llvm_value_type);
                auto reduced = [&] {
                    switch (op) {
                        case xir::ThreadGroupOp::WARP_ACTIVE_MAX:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateBinaryIntrinsic(llvm::Intrinsic::smax, x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateBinaryIntrinsic(llvm::Intrinsic::umax, x, y) :
                                                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_MIN:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateBinaryIntrinsic(llvm::Intrinsic::smin, x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateBinaryIntrinsic(llvm::Intrinsic::umin, x, y) :
                                                                            b.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWMul(x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateMul(x, y) :
                                                                            b.CreateFMul(x, y);
                        case xir::ThreadGroupOp::WARP_ACTIVE_SUM:
                            return inst->type()->is_int_or_int_vector()   ? b.CreateNSWAdd(x, y) :
                                   inst->type()->is_uint_or_uint_vector() ? b.CreateAdd(x, y) :
                                                                            b.CreateFAdd(x, y);
                        default: break;
                    }
                    LUISA_ERROR_WITH_LOCATION("Invalid warp reduction op.");
                }();
                return pack_into_i32_vector(reduced).first;
            });
            return unpack_from_i32_vector(llvm_result_packed, llvm_value_type);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ballot, {llvm_pred});
            return b.CreateICmpEQ(llvm_ballot, llvm_active_mask);
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: {
            LUISA_DEBUG_ASSERT(inst->type()->is_bool());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ballot, {llvm_pred});
            return b.CreateICmpNE(llvm_ballot, b.getInt32(0));
        }
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: {
            LUISA_DEBUG_ASSERT(inst->type() == Type::of<int4>() || inst->type() == Type::of<uint4>());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ballot, {llvm_pred});
            auto llvm_zero = llvm::Constant::getNullValue(llvm::VectorType::get(b.getInt32Ty(), 4, false));
            return b.CreateInsertElement(llvm_zero, llvm_ballot, b.getInt64(0));
        }
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: {
            LUISA_DEBUG_ASSERT(inst->type()->is_int32() || inst->type()->is_uint32());
            auto llvm_pred = _get_llvm_value(b, func_ctx, inst->operand(0));
            LUISA_DEBUG_ASSERT(llvm_pred->getType()->isIntegerTy(1));
            auto llvm_ballot = b.CreateIntrinsic(b.getInt32Ty(), llvm::Intrinsic::amdgcn_ballot, {llvm_pred});
            auto llvm_prefix_mask = _read_warp_prefix_lane_mask(b, func_ctx);
            auto llvm_ballot_and_prefix = b.CreateAnd(llvm_ballot, llvm_prefix_mask);
            return b.CreateUnaryIntrinsic(llvm::Intrinsic::ctpop, llvm_ballot_and_prefix);
        }
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: {
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_value_type = llvm_value->getType();
            LUISA_DEBUG_ASSERT(llvm_value_type->isIntOrIntVectorTy() || llvm_value_type->isFPOrFPVectorTy());
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_lane_id = _read_warp_lane_id(b, func_ctx);
            auto llvm_packed_value = pack_into_i32_vector(llvm_value).first;
            auto llvm_result_packed = static_cast<llvm::Value *>(nullptr);
            if (op == xir::ThreadGroupOp::WARP_PREFIX_SUM) {
                auto llvm_unit = pack_into_i32_vector(llvm::Constant::getNullValue(llvm_value_type)).first;
                llvm_result_packed = reduce_prefix(llvm_active_mask, llvm_lane_id, llvm_unit, llvm_packed_value, [&](auto x, auto y) noexcept {
                    x = unpack_from_i32_vector(x, llvm_value_type);
                    y = unpack_from_i32_vector(y, llvm_value_type);
                    auto result = inst->type()->is_int_or_int_vector()   ? b.CreateNSWAdd(x, y) :
                                  inst->type()->is_uint_or_uint_vector() ? b.CreateAdd(x, y) :
                                                                           b.CreateFAdd(x, y);
                    return pack_into_i32_vector(result).first;
                });
            } else if (op == xir::ThreadGroupOp::WARP_PREFIX_PRODUCT) {
                auto llvm_unit = pack_into_i32_vector(
                                     llvm_value_type->isIntOrIntVectorTy() ?
                                         llvm::ConstantInt::get(llvm_value_type, 1) :
                                         llvm::ConstantFP::get(llvm_value_type, 1.))
                                     .first;
                llvm_result_packed = reduce_prefix(llvm_active_mask, llvm_lane_id, llvm_unit, llvm_packed_value, [&](auto x, auto y) noexcept {
                    x = unpack_from_i32_vector(x, llvm_value_type);
                    y = unpack_from_i32_vector(y, llvm_value_type);
                    auto result = inst->type()->is_int_or_int_vector()   ? b.CreateNSWMul(x, y) :
                                  inst->type()->is_uint_or_uint_vector() ? b.CreateMul(x, y) :
                                                                           b.CreateFMul(x, y);
                    return pack_into_i32_vector(result).first;
                });
            } else {
                LUISA_ERROR_WITH_LOCATION("Invalid warp prefix op.");
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value_type);
        }
        case xir::ThreadGroupOp::WARP_READ_LANE: [[fallthrough]];
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: {
            auto llvm_active_mask = _read_warp_active_lane_mask(b);
            auto llvm_value = _get_llvm_value(b, func_ctx, inst->operand(0));
            auto llvm_lane_id = op == xir::ThreadGroupOp::WARP_READ_LANE ?
                                    _get_llvm_value(b, func_ctx, inst->operand(1)) :
                                    b.CreateBinaryIntrinsic(llvm::Intrinsic::cttz, llvm_active_mask, b.getInt1(true));
            LUISA_DEBUG_ASSERT(llvm_lane_id->getType()->isIntegerTy(32));
            auto [llvm_value_packed, llvm_packed_i32_count] = pack_into_i32_vector(llvm_value);
            auto llvm_result_packed = static_cast<llvm::Value *>(llvm::PoisonValue::get(llvm_value_packed->getType()));
            for (auto i = 0; i < llvm_packed_i32_count; i++) {
                auto llvm_local_elem = b.CreateExtractElement(llvm_value_packed, i);
                auto llvm_elem_from_lane = shuffle_idx(llvm_local_elem, llvm_lane_id);
                llvm_result_packed = b.CreateInsertElement(llvm_result_packed, llvm_elem_from_lane, i);
            }
            return unpack_from_i32_vector(llvm_result_packed, llvm_value->getType());
        }
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: {
            return b.CreateIntrinsic(b.getVoidTy(), llvm::Intrinsic::amdgcn_s_barrier, {});
        }
    }
    LUISA_NOT_IMPLEMENTED();
}

}// namespace luisa::compute::hip
