//
// Created by mike on 4/1/25.
//

#ifdef LUISA_ENABLE_XIR

#include <luisa/core/stl/algorithm.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/metadata/name.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/metadata/location.h>

#include "cuda_codegen_xir.h"

namespace luisa::compute::cuda {

CUDACodegenXIR::CUDACodegenXIR(StringScratch &scratch, bool allow_indirect) noexcept
    : _scratch{scratch},
      _allow_indirect_dispatch{allow_indirect},
      _ray_type{Type::of<Ray>()},
      _triangle_hit_type{Type::of<TriangleHit>()},
      _procedural_hit_type{Type::of<ProceduralHit>()},
      _committed_hit_type{Type::of<CommittedHit>()},
      _ray_query_all_type{Type::of<RayQueryAll>()},
      _ray_query_any_type{Type::of<RayQueryAny>()},
      _indirect_buffer_type{Type::of<IndirectDispatchBuffer>()},
      _motion_srt_type{Type::of<MotionInstanceTransformSRT>()} {}

CUDACodegenXIR::~CUDACodegenXIR() noexcept = default;

bool CUDACodegenXIR::_should_emit_global_constant(const xir::Constant *c) noexcept {
    return !c->type()->is_basic();
}

bool CUDACodegenXIR::_is_builtin_type(const Type *t) const noexcept {
    return t == _ray_type ||
           t == _triangle_hit_type ||
           t == _procedural_hit_type ||
           t == _committed_hit_type ||
           t == _ray_query_all_type ||
           t == _ray_query_any_type ||
           t == _indirect_buffer_type ||
           t == _motion_srt_type;
}

void CUDACodegenXIR::_emit_type_name(const Type *type) noexcept {
    if (type == nullptr) {
        _scratch << "void";
        return;
    }
    switch (type->tag()) {
        case Type::Tag::BOOL: _scratch << "lc_bool"; break;
        case Type::Tag::FLOAT16: _scratch << "lc_ushort"; break;
        case Type::Tag::FLOAT32: _scratch << "lc_uint"; break;
        case Type::Tag::FLOAT64: _scratch << "lc_ulong"; break;
        case Type::Tag::INT8: _scratch << "lc_byte"; break;
        case Type::Tag::UINT8: _scratch << "lc_ubyte"; break;
        case Type::Tag::INT16: _scratch << "lc_short"; break;
        case Type::Tag::UINT16: _scratch << "lc_ushort"; break;
        case Type::Tag::INT32: _scratch << "lc_int"; break;
        case Type::Tag::UINT32: _scratch << "lc_uint"; break;
        case Type::Tag::INT64: _scratch << "lc_long"; break;
        case Type::Tag::UINT64: _scratch << "lc_ulong"; break;
        case Type::Tag::VECTOR: {
            _emit_type_name(type->element());
            _scratch << type->dimension();
            break;
        }
        case Type::Tag::MATRIX: {
            auto dim = type->dimension();
            _scratch << "lc_float" << dim << "x" << dim;
            break;
        }
        case Type::Tag::ARRAY: {
            _scratch << "lc_array<";
            _emit_type_name(type->element());
            _scratch << ", ";
            _scratch << type->dimension() << ">";
            break;
        }
        case Type::Tag::STRUCTURE: {
            if (type == _ray_type) {
                _scratch << "LCRay";
            } else if (type == _triangle_hit_type) {
                _scratch << "LCTriangleHit";
            } else if (type == _procedural_hit_type) {
                _scratch << "LCProceduralHit";
            } else if (type == _committed_hit_type) {
                _scratch << "LCCommittedHit";
            } else if (type == _motion_srt_type) {
                _scratch << "LCMotionSRT";
            } else {
                _scratch << "S" << hash_to_string(type->hash());
            }
            break;
        }
        case Type::Tag::CUSTOM: {
            if (type == _ray_query_all_type) {
                _scratch << "LCRayQueryAll";
            } else if (type == _ray_query_any_type) {
                _scratch << "LCRayQueryAny";
            } else if (type == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer";
            } else {
                LUISA_ERROR_WITH_LOCATION(
                    "Unsupported custom type: {}.",
                    type->description());
            }
            break;
        }
        case Type::Tag::BUFFER: {
            if (type == _indirect_buffer_type) {
                _scratch << "LCIndirectBuffer";
            } else {
                _scratch << "LCBuffer<";
                if (auto elem = type->element()) {
                    _emit_type_name(elem);
                } else {
                    _scratch << "lc_ubyte";
                }
                _scratch << ">";
            }
            break;
        }
        case Type::Tag::TEXTURE: {
            auto elem = type->element();
            auto dim = type->dimension();
            _scratch << "LCTexture" << dim << "D<";
            _emit_type_name(elem);
            _scratch << ">";
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: _scratch << "LCBindlessArray"; break;
        case Type::Tag::ACCEL: _scratch << "LCAccel"; break;
        default: LUISA_ERROR_WITH_LOCATION(
            "Invalid type {} in CUDA codegen.",
            type->description());
    }
}

void CUDACodegenXIR::_emit_type_definition(const Type *type, luisa::unordered_set<const Type *> &defined_types) noexcept {
    if (!defined_types.emplace(type).second) { return; }
    LUISA_DEBUG_ASSERT(type->tag() == Type::Tag::STRUCTURE, "Type is not a structure.");
    // process the members
    auto members = type->members();
    for (auto m : members) {
        if (m->is_structure()) {
            _emit_type_definition(m, defined_types);
        }
    }
    // generate the type definition if not a builtin type
    if (!_is_builtin_type(type)) {
        _scratch << "struct alignas(" << type->alignment() << ") ";
        _emit_type_name(type);
        _scratch << " {\n";
        for (auto i = 0u; i < members.size(); i++) {
            _scratch << "  ";
            _emit_type_name(members[i]);
            _scratch << " m" << i << ";\n";
        }
        _scratch << "};\n\n";
    }
}

void CUDACodegenXIR::_emit_type_definitions(luisa::unordered_set<const Type *> used_types) noexcept {
    luisa::vector<const Type *> types;
    types.reserve(used_types.size());
    for (auto t : used_types) {
        if (t != nullptr && t->is_structure()) {
            types.emplace_back(t);
        }
    }
    luisa::sort(types.begin(), types.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });
    used_types.clear();
    for (auto t : types) {
        _emit_type_definition(t, used_types);
    }
}

namespace {

template<typename T>
[[nodiscard]] auto cuda_codegen_xir_decode_literal(const void *data) noexcept {
    return *static_cast<const T *>(data);
}

void cuda_codegen_xir_emit_literal(StringScratch &s, const Type *type, const std::byte *data) noexcept {
    switch (type->tag()) {
        case Type::Tag::BOOL: s << (cuda_codegen_xir_decode_literal<bool>(data) ? "true" : "false"); break;
        case Type::Tag::INT8: s << luisa::format("lc_byte({})", static_cast<int>(cuda_codegen_xir_decode_literal<int8_t>(data))); break;
        case Type::Tag::UINT8: s << luisa::format("lc_ubyte({})", static_cast<uint>(cuda_codegen_xir_decode_literal<uint8_t>(data))); break;
        case Type::Tag::INT16: s << luisa::format("lc_short({})", cuda_codegen_xir_decode_literal<int16_t>(data)); break;
        case Type::Tag::UINT16: s << luisa::format("lc_ushort({})", cuda_codegen_xir_decode_literal<uint16_t>(data)); break;
        case Type::Tag::INT32: s << luisa::format("lc_int({})", cuda_codegen_xir_decode_literal<int32_t>(data)); break;
        case Type::Tag::UINT32: s << luisa::format("lc_uint({})", cuda_codegen_xir_decode_literal<uint32_t>(data)); break;
        case Type::Tag::INT64: s << luisa::format("lc_long({})", cuda_codegen_xir_decode_literal<int64_t>(data)); break;
        case Type::Tag::UINT64: s << luisa::format("lc_ulong({})", cuda_codegen_xir_decode_literal<uint64_t>(data)); break;
        case Type::Tag::FLOAT16: {
            auto v = cuda_codegen_xir_decode_literal<half>(data);
            LUISA_ASSERT(!luisa::isnan(v), "Encountered with NaN.");
            if (luisa::isinf(v)) {
                s << luisa::format("(lc_bit_cast<lc_half, lc_ushort>(0x{:04x}u))", luisa::bit_cast<ushort>(v));
            } else {
                s << luisa::format("lc_half({})", static_cast<float>(v));
            }
            break;
        }
        case Type::Tag::FLOAT32: {
            auto v = cuda_codegen_xir_decode_literal<float>(data);
            LUISA_ASSERT(!luisa::isnan(v), "Encountered with NaN.");
            if (luisa::isinf(v)) {
                s << luisa::format("(lc_bit_cast<lc_float, lc_uint>(0x{:08x}u))", luisa::bit_cast<uint>(v));
            } else {
                s << luisa::format("lc_float({})", v);
            }
            break;
        }
        case Type::Tag::FLOAT64: {
            auto v = cuda_codegen_xir_decode_literal<double>(data);
            LUISA_ASSERT(!luisa::isnan(v), "Encountered with NaN.");
            if (luisa::isinf(v)) {
                s << luisa::format("(lc_bit_cast<lc_double, lc_ulong>(0x{:016x}ull))", luisa::bit_cast<ulong>(v));
            } else {
                s << luisa::format("lc_double({})", v);
            }
            break;
        }
        case Type::Tag::VECTOR: {
            auto elem = type->element();
            auto elem_stride = elem->size();
            auto dim = type->dimension();
            s << "lc_make_" << elem->description() << dim << "(";
            for (auto i = 0u; i < dim; i++) {
                cuda_codegen_xir_emit_literal(s, elem, data + i * elem_stride);
                if (i != dim - 1u) { s << ", "; }
            }
            s << ")";
            break;
        }
        case Type::Tag::MATRIX: {
            auto elem = type->element();
            auto dim = type->dimension();
            auto col = Type::vector(elem, dim);
            auto col_stride = col->size();
            s << "lc_make_" << elem->description() << dim << "x" << dim << "(";
            for (auto i = 0u; i < dim; i++) {
                cuda_codegen_xir_emit_literal(s, col, data + i * col_stride);
                if (i != dim - 1u) { s << ", "; }
            }
            s << ")";
            break;
        }
        default: LUISA_NOT_IMPLEMENTED("Unsupported literal type {}.", type->description());
    }
}

}// namespace

void CUDACodegenXIR::_emit_value_name(const xir::Value *value, bool is_use) noexcept {
    auto get_index = [](const xir::Value *v, luisa::unordered_map<const xir::Value *, size_t> &indices) noexcept {
        return indices.try_emplace(v, indices.size()).first->second;
    };
    auto get_global_index = [&](const xir::Value *v) noexcept {
        return get_index(v, _global_value_indices);
    };
    auto get_local_index = [&](const xir::Value *v) noexcept {
        return get_index(v, _local_value_indices);
    };
    LUISA_DEBUG_ASSERT(value != nullptr, "Value is null.");
    switch (value->derived_value_tag()) {
        case xir::DerivedValueTag::UNDEFINED: {
            _scratch << "(lc_undef<";
            _emit_type_name(value->type());
            _scratch << ">())";
            break;
        }
        case xir::DerivedValueTag::FUNCTION: {
            switch (static_cast<const xir::Function *>(value)->derived_function_tag()) {
                case xir::DerivedFunctionTag::KERNEL: _scratch << "kernel_main"; break;
                case xir::DerivedFunctionTag::CALLABLE: _scratch << "callable_" << get_global_index(value); break;
                case xir::DerivedFunctionTag::EXTERNAL: LUISA_NOT_IMPLEMENTED(
                    "External function {} is not supported in XIR-based CUDA codegen.",
                    value->name().value_or("unknown"));
            }
            break;
        }
        case xir::DerivedValueTag::BASIC_BLOCK: LUISA_ERROR_WITH_LOCATION("Cannot emit name for basic block.");
        case xir::DerivedValueTag::INSTRUCTION: {
            _scratch << (value->is_lvalue() ? "pv" : "v")
                     << get_local_index(value);
            break;
        }
        case xir::DerivedValueTag::CONSTANT: {
            if (auto c = static_cast<const xir::Constant *>(value); _should_emit_global_constant(c)) {
                if (is_use) {
                    _scratch << "(lc_decode_bytes<";
                    _emit_type_name(c->type());
                    _scratch << ">(";
                }
                _scratch << "c" << get_global_index(value);
                if (is_use) {
                    _scratch << "))";
                }
            } else {// literal
                cuda_codegen_xir_emit_literal(_scratch, c->type(), static_cast<const std::byte *>(c->data()));
            }
            break;
        }
        case xir::DerivedValueTag::ARGUMENT: {
            _scratch << (value->is_lvalue() ? "pa" : "a")
                     << get_local_index(value);
            break;
        }
        case xir::DerivedValueTag::SPECIAL_REGISTER: {
            switch (static_cast<const xir::SpecialRegister *>(value)->derived_special_register_tag()) {
                case xir::DerivedSpecialRegisterTag::THREAD_ID: _scratch << "sreg_tid"; break;
                case xir::DerivedSpecialRegisterTag::BLOCK_ID: _scratch << "sreg_bid"; break;
                case xir::DerivedSpecialRegisterTag::WARP_LANE_ID: _scratch << "sreg_lid"; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_ID: _scratch << "sreg_did"; break;
                case xir::DerivedSpecialRegisterTag::KERNEL_ID: _scratch << "sreg_kid"; break;
                case xir::DerivedSpecialRegisterTag::OBJECT_ID: LUISA_NOT_IMPLEMENTED("Object ID is not supported in XIR-based CUDA codegen.");
                case xir::DerivedSpecialRegisterTag::BLOCK_SIZE: _scratch << "sreg_bs"; break;
                case xir::DerivedSpecialRegisterTag::WARP_SIZE: _scratch << "sreg_ws"; break;
                case xir::DerivedSpecialRegisterTag::DISPATCH_SIZE: _scratch << "sreg_ls"; break;
            }
            break;
        }
    }
}

void CUDACodegenXIR::_emit_global_constants(luisa::unordered_set<const xir::Constant *> used_constants) noexcept {
    luisa::vector<const xir::Constant *> constants;
    constants.reserve(used_constants.size());
    for (auto &&c : used_constants) {
        if (_should_emit_global_constant(c)) {
            constants.emplace_back(c);
        }
    }
    luisa::sort(constants.begin(), constants.end(), [](auto a, auto b) noexcept {
        return a->hash() < b->hash();
    });
    for (auto c : constants) {
        _scratch << "__constant__ LC_CONSTANT lc_ubyte ";
        _emit_value_name(c, false);
        auto n = c->type()->size();
        _scratch << "[" << n << "] = /* ";
        _emit_type_name(c->type());
        _scratch << " */ {";
        auto bytes = static_cast<const std::uint8_t *>(c->data());
        for (auto i = 0u; i < n; i++) {
            if (i % 16u == 0u) { _scratch << "\n   "; }
            _scratch << luisa::format(" 0x{:02x},", static_cast<uint32_t>(bytes[i]));
        }
        _scratch << "\n};\n\n";
    }
}

void CUDACodegenXIR::_emit_result_value_eq(const xir::Instruction *inst) noexcept {
    if (auto ret_type = inst->type()) {
        _emit_type_name(ret_type);
        if (inst->is_lvalue()) {
            _scratch << " *const ";
        } else {
            _scratch << " const ";
        }
        _emit_value_name(inst);
        _scratch << " = ";
    }
}

void CUDACodegenXIR::_emit_instructions(const xir::InstructionList &inst_list, int indent) noexcept {
    for (auto &&inst : inst_list) {
        _emit_metadata(inst.metadata_list(), indent);
        _emit_indent(indent);
        auto emit_result_value_eq = [&] { _emit_result_value_eq(&inst); };
        switch (inst.derived_instruction_tag()) {
            case xir::DerivedInstructionTag::IF: {
                _with_control_flow(&inst, [&] {
                    _emit_if_inst(static_cast<const xir::IfInst *>(&inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::SWITCH: {
                _with_control_flow(&inst, [&] {
                    _emit_switch_inst(static_cast<const xir::SwitchInst *>(&inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::LOOP: {
                _with_control_flow(&inst, [&] {
                    _emit_loop_inst(static_cast<const xir::LoopInst *>(&inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::SIMPLE_LOOP: {
                _with_control_flow(&inst, [&] {
                    _emit_simple_loop_inst(static_cast<const xir::SimpleLoopInst *>(&inst), indent);
                });
                break;
            }
            case xir::DerivedInstructionTag::BRANCH: _emit_branch_inst(static_cast<const xir::BranchInst *>(&inst)); break;
            case xir::DerivedInstructionTag::CONDITIONAL_BRANCH: _emit_conditional_branch_inst(static_cast<const xir::ConditionalBranchInst *>(&inst)); break;
            case xir::DerivedInstructionTag::UNREACHABLE: {
                if (auto &&msg = static_cast<const xir::UnreachableInst *>(&inst)->message(); !msg.empty()) {
                    _scratch << "lc_unreachable_with_message(__FILE__, __LINE__, "
                             << luisa::format("{:?}", msg)
                             << ");";
                } else {
                    _scratch << "lc_unreachable(__FILE__, __LINE__);";
                }
                break;
            }
            case xir::DerivedInstructionTag::BREAK: {
                LUISA_ASSERT(!_control_flow_stack.empty(), "Control flow stack is empty.");
                if (_control_flow_stack.back()->isa<xir::LoopInst>()) {
                    _scratch << "loop_break = true;\n";
                    _emit_indent(indent);
                }
                _scratch << "break;";
                break;
            }
            case xir::DerivedInstructionTag::CONTINUE: {
                LUISA_ASSERT(!_control_flow_stack.empty(), "Control flow stack is empty.");
                if (_control_flow_stack.back()->isa<xir::LoopInst>()) {
                    _scratch << "break;";
                } else {
                    _scratch << "continue;";
                }
                break;
            }
            case xir::DerivedInstructionTag::RETURN: {
                if (auto ret = static_cast<const xir::ReturnInst *>(&inst)->return_value()) {
                    _scratch << "return ";
                    _emit_value_name(ret);
                    _scratch << ";";
                } else {
                    _scratch << "return;";
                }
                break;
            }
            case xir::DerivedInstructionTag::RASTER_DISCARD: LUISA_NOT_IMPLEMENTED("Raster discard is not supported in XIR-based CUDA codegen.");
            case xir::DerivedInstructionTag::PHI: LUISA_ERROR_WITH_LOCATION("Phi instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::ALLOCA: {
                // emit the alloca variable
                if (static_cast<const xir::AllocaInst *>(&inst)->is_shared()) {
                    _scratch << "__shared__ ";
                }
                _emit_type_name(inst.type());
                _scratch << " ";
                _emit_value_name(&inst);
                _scratch << "_alloca;";
                // emit the pointer to the alloca variable
                _scratch << "\n";
                _emit_indent(indent);
                emit_result_value_eq();
                _scratch << "&";
                _emit_value_name(&inst);
                _scratch << "_alloca;";
                break;
            }
            case xir::DerivedInstructionTag::LOAD: {
                emit_result_value_eq();
                _scratch << "*(";
                _emit_value_name(static_cast<const xir::LoadInst *>(&inst)->variable());
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::STORE: {
                _scratch << "*(";
                _emit_value_name(static_cast<const xir::StoreInst *>(&inst)->variable());
                _scratch << ") = ";
                _emit_value_name(static_cast<const xir::StoreInst *>(&inst)->value());
                _scratch << ";";
                break;
            }
            case xir::DerivedInstructionTag::GEP: _emit_gep_inst(static_cast<const xir::GEPInst *>(&inst)); break;
            case xir::DerivedInstructionTag::ATOMIC: _emit_atomic_inst(static_cast<const xir::AtomicInst *>(&inst)); break;
            case xir::DerivedInstructionTag::ARITHMETIC: _emit_arithmetic_inst(static_cast<const xir::ArithmeticInst *>(&inst)); break;
            case xir::DerivedInstructionTag::THREAD_GROUP: _emit_thread_group_inst(static_cast<const xir::ThreadGroupInst *>(&inst)); break;
            case xir::DerivedInstructionTag::RESOURCE_QUERY: _emit_resource_query_inst(static_cast<const xir::ResourceQueryInst *>(&inst)); break;
            case xir::DerivedInstructionTag::RESOURCE_READ: _emit_resource_read_inst(static_cast<const xir::ResourceReadInst *>(&inst)); break;
            case xir::DerivedInstructionTag::RESOURCE_WRITE: _emit_resource_write_inst(static_cast<const xir::ResourceWriteInst *>(&inst)); break;
            case xir::DerivedInstructionTag::RAY_QUERY_LOOP: LUISA_ERROR_WITH_LOCATION("Ray query loop instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::RAY_QUERY_DISPATCH: LUISA_ERROR_WITH_LOCATION("Ray query dispatch instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_READ: _emit_ray_query_object_read_inst(static_cast<const xir::RayQueryObjectReadInst *>(&inst)); break;
            case xir::DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: _emit_ray_query_object_write_inst(static_cast<const xir::RayQueryObjectWriteInst *>(&inst)); break;
            case xir::DerivedInstructionTag::RAY_QUERY_PIPELINE: LUISA_NOT_IMPLEMENTED("Ray query pipeline has not been implemented in XIR-based CUDA codegen.");
            case xir::DerivedInstructionTag::AUTODIFF_SCOPE: LUISA_ERROR_WITH_LOCATION("Autodiff scope instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::AUTODIFF_INTRINSIC: LUISA_ERROR_WITH_LOCATION("Autodiff intrinsic instructions should be eliminated before codegen.");
            case xir::DerivedInstructionTag::CALL: {
                emit_result_value_eq();
                auto call = static_cast<const xir::CallInst *>(&inst);
                _emit_value_name(call->callee());
                _scratch << "(";
                auto any_arg = false;
                for (auto &&arg_use : call->argument_uses()) {
                    any_arg = true;
                    _emit_value_name(arg_use->value());
                    _scratch << ", ";
                }
                if (!_requires_optix && _requires_printing) {
                    any_arg = true;
                    _scratch << "print_buffer, ";
                }
                if (any_arg) {
                    _scratch.pop_back();
                    _scratch.pop_back();
                }
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::CAST: {
                emit_result_value_eq();
                auto cast = static_cast<const xir::CastInst *>(&inst);
                switch (cast->op()) {
                    case xir::CastOp::STATIC_CAST: _scratch << "static_cast<"; break;
                    case xir::CastOp::BITWISE_CAST: _scratch << "lc_bit_cast<"; break;
                }
                _emit_type_name(cast->type());
                _scratch << ">(";
                _emit_value_name(cast->value());
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::PRINT: {
                auto p = static_cast<const xir::PrintInst *>(&inst);
                auto info = _print_info.at(p);
                _scratch << "lc_print(LC_PRINT_BUFFER, ";
                _emit_type_name(info.type);
                _scratch << "{" << info.type->size() << ", "
                         << info.index;
                for (auto op_use : p->operand_uses()) {
                    _scratch << ", ";
                    _emit_value_name(op_use->value());
                }
                _scratch << "});";
                break;
            }
            case xir::DerivedInstructionTag::CLOCK: {
                emit_result_value_eq();
                _scratch << "clock64();";
                break;
            }
            case xir::DerivedInstructionTag::ASSERT: {
                if (auto a = static_cast<const xir::AssertInst *>(&inst); a->message().empty()) {
                    _scratch << "lc_assert(";
                    _emit_value_name(a->condition());
                    _scratch << ");";
                } else {
                    _scratch << "lc_assert_with_message(";
                    _emit_value_name(a->condition());
                    _scratch << ", "
                             << luisa::format("{:?}", a->message())
                             << ");";
                }
                break;
            }
            case xir::DerivedInstructionTag::ASSUME: {
                _scratch << "lc_assume(";
                _emit_value_name(static_cast<const xir::AssumeInst *>(&inst)->condition());
                _scratch << ");";
                break;
            }
            case xir::DerivedInstructionTag::OUTLINE: LUISA_ERROR_WITH_LOCATION("Outline instructions should be eliminated before codegen.");
        }
        _scratch << "\n";
        if (auto merge = inst.control_flow_merge(); merge != nullptr && merge->merge_block() != nullptr) {
            _emit_instructions(merge->merge_block()->instructions(), indent);
        }
    }
}

void CUDACodegenXIR::_emit_metadata(const xir::MetadataList &md_list, int indent) const noexcept {
    for (auto &&md : md_list) {
        _emit_indent(indent);
        _scratch << "// ";
        switch (md.derived_metadata_tag()) {
            case xir::DerivedMetadataTag::NAME: {
                auto name_md = static_cast<const xir::NameMD *>(&md);
                _scratch << luisa::format("name: {:?}",
                                          name_md->name());
                break;
            }
            case xir::DerivedMetadataTag::LOCATION: {
                auto loc_md = static_cast<const xir::LocationMD *>(&md);
                _scratch << luisa::format("location: {:?}, line {}",
                                          loc_md->file().string(), loc_md->line());
                break;
            }
            case xir::DerivedMetadataTag::COMMENT: {
                auto comment_md = static_cast<const xir::CommentMD *>(&md);
                for (auto c : comment_md->comment()) {
                    if (c == '\n') {
                        _scratch << "\n";
                        _emit_indent(indent);
                        _scratch << "// ";
                    } else if (isprint(c)) {
                        char s[] = {c, '\0'};
                        _scratch << s;
                    }
                }
                break;
            }
            case xir::DerivedMetadataTag::CURVE_BASIS: {
                auto bases = static_cast<const xir::CurveBasisMD *>(&md)->curve_basis_set();
                _scratch << "curve basis:";
                if (bases.none()) {
                    _scratch << " none";
                } else {
                    if (bases.test(CurveBasis::PIECEWISE_LINEAR)) { _scratch << " piecewise_linear"; }
                    if (bases.test(CurveBasis::CUBIC_BSPLINE)) { _scratch << " cubic_bspline"; }
                    if (bases.test(CurveBasis::CATMULL_ROM)) { _scratch << " catmull_rom"; }
                    if (bases.test(CurveBasis::BEZIER)) { _scratch << " bezier"; }
                }
                break;
            }
        }
        _scratch << "\n";
    }
}

void CUDACodegenXIR::_emit_indent(int indent) const noexcept {
    _scratch.string().append(indent * 2, ' ');
}

void CUDACodegenXIR::_emit_if_inst(const xir::IfInst *inst, int indent) noexcept {
    _scratch << "if (";
    _emit_value_name(inst->condition());
    _scratch << ") {";
    if (auto true_block = inst->true_block(); true_block != nullptr && !true_block->instructions().empty()) {
        _scratch << "\n";
        _emit_instructions(true_block->instructions(), indent + 1);
        _emit_indent(indent);
    }
    _scratch << "} else {";
    if (auto false_block = inst->false_block(); false_block != nullptr && !false_block->instructions().empty()) {
        _scratch << "\n";
        _emit_instructions(false_block->instructions(), indent + 1);
        _emit_indent(indent);
    }
    _scratch << "}";
}

void CUDACodegenXIR::_emit_switch_inst(const xir::SwitchInst *inst, int indent) noexcept {
    _scratch << "switch (";
    _emit_value_name(inst->value());
    _scratch << ") {";
    auto value_type = inst->value()->type();
    for (auto i = 0u; i < inst->case_count(); i++) {
        _scratch << "\n";
        _emit_indent(indent + 1);
        _scratch << "case static_cast<";
        _emit_type_name(value_type);
        _scratch << ">(" << inst->case_value(i) << "): {\n";
        if (auto block = inst->case_block(i); block != nullptr) {
            _emit_instructions(block->instructions(), indent + 2);
        }
        _emit_indent(indent + 1);
        _scratch << "}";
    }
    if (auto default_block = inst->default_block(); default_block != nullptr) {
        _scratch << "\n";
        _emit_indent(indent + 1);
        _scratch << "default: {\n";
        _emit_instructions(default_block->instructions(), indent + 2);
        _emit_indent(indent + 1);
        _scratch << "}\n";
    }
}

void CUDACodegenXIR::_emit_loop_inst(const xir::LoopInst *inst, int indent) noexcept {
    // template:
    // loop {
    //     loop_break = false;
    //     prepare();
    //     do {
    //         body {
    //             // break => { loop_break = true; break; }
    //             // continue => { break; }
    //         }
    //     } while (false);
    //     if (loop_break) break;
    //     update();
    // }
    _scratch << "for (;;) { /* generic loop */\n";
    _emit_indent(indent + 1);
    _scratch << "bool loop_break = false;\n";
    _emit_indent(indent + 1);
    _scratch << "/* generic loop prepare */\n";
    // prepare
    if (auto prepare_block = inst->prepare_block(); prepare_block != nullptr && !prepare_block->instructions().empty()) {
        _emit_instructions(prepare_block->instructions(), indent + 1);
    }
    _emit_indent(indent + 1);
    // body
    _scratch << "do { /* generic loop body */";
    if (auto body_block = inst->body_block(); body_block != nullptr && !body_block->instructions().empty()) {
        _scratch << "\n";
        _emit_instructions(body_block->instructions(), indent + 2);
        _emit_indent(indent + 1);
    }
    _scratch << "} while (false);\n";
    // break
    _emit_indent(indent + 1);
    _scratch << "if (loop_break) { break; }\n";
    _emit_indent(indent + 1);
    _scratch << "/* generic loop update */\n";
    // update
    if (auto update_block = inst->update_block(); update_block != nullptr && !update_block->instructions().empty()) {
        _emit_instructions(update_block->instructions(), indent + 1);
    }
    _emit_indent(indent);
    _scratch << "}";
}

void CUDACodegenXIR::_emit_simple_loop_inst(const xir::SimpleLoopInst *inst, int indent) noexcept {
    // do-while loop
    _scratch << "for (;;) { /* simple loop */";
    if (auto body_block = inst->body_block(); body_block != nullptr && !body_block->instructions().empty()) {
        _scratch << "\n";
        _emit_instructions(body_block->instructions(), indent + 1);
        _emit_indent(indent);
    }
    _scratch << "}";
}

void CUDACodegenXIR::_emit_intrinsic_call(luisa::string_view name, const xir::Instruction *inst) noexcept {
    _emit_result_value_eq(inst);
    _scratch << name << "(";
    auto any_arg = false;
    for (auto &&arg_use : inst->operand_uses()) {
        any_arg = true;
        _emit_value_name(arg_use->value());
        _scratch << ", ";
    }
    if (any_arg) {
        _scratch.pop_back();
        _scratch.pop_back();
    }
    _scratch << ");";
}

void CUDACodegenXIR::_emit_gep_inst(const xir::GEPInst *inst) noexcept {
}

void CUDACodegenXIR::_emit_atomic_inst(const xir::AtomicInst *inst) noexcept {
}

void CUDACodegenXIR::_emit_arithmetic_inst(const xir::ArithmeticInst *inst) noexcept {
    auto u = [&](auto op) noexcept { _emit_with_template(inst, op, "(", 0, ")"); };
    auto b = [&](auto op) noexcept { _emit_with_template(inst, "(", 0, ") ", op, " (", 1, ")"); };
    auto f = [&](auto s) noexcept { _emit_intrinsic_call(s, inst); };
    switch (inst->op()) {
        case xir::ArithmeticOp::UNARY_PLUS: u("+"); break;
        case xir::ArithmeticOp::UNARY_MINUS: u("-"); break;
        case xir::ArithmeticOp::UNARY_BIT_NOT: u("~"); break;
        case xir::ArithmeticOp::BINARY_ADD: b("+"); break;
        case xir::ArithmeticOp::BINARY_SUB: b("-"); break;
        case xir::ArithmeticOp::BINARY_MUL: b("*"); break;
        case xir::ArithmeticOp::BINARY_DIV: b("/"); break;
        case xir::ArithmeticOp::BINARY_MOD: b("%"); break;
        case xir::ArithmeticOp::BINARY_BIT_AND: b("&"); break;
        case xir::ArithmeticOp::BINARY_BIT_OR: b("|"); break;
        case xir::ArithmeticOp::BINARY_BIT_XOR: b("^"); break;
        case xir::ArithmeticOp::BINARY_SHIFT_LEFT: b("<<"); break;
        case xir::ArithmeticOp::BINARY_SHIFT_RIGHT: b(">>"); break;
        case xir::ArithmeticOp::BINARY_ROTATE_LEFT: LUISA_NOT_IMPLEMENTED("lc_rotl"); break;
        case xir::ArithmeticOp::BINARY_ROTATE_RIGHT: LUISA_NOT_IMPLEMENTED("lc_rotr"); break;
        case xir::ArithmeticOp::BINARY_LESS: b("<"); break;
        case xir::ArithmeticOp::BINARY_GREATER: b(">"); break;
        case xir::ArithmeticOp::BINARY_LESS_EQUAL: b("<="); break;
        case xir::ArithmeticOp::BINARY_GREATER_EQUAL: b(">="); break;
        case xir::ArithmeticOp::BINARY_EQUAL: b("=="); break;
        case xir::ArithmeticOp::BINARY_NOT_EQUAL: b("!="); break;
        case xir::ArithmeticOp::ALL: f("lc_all"); break;
        case xir::ArithmeticOp::ANY: f("lc_any"); break;
        case xir::ArithmeticOp::SELECT: f("lc_select"); break;
        case xir::ArithmeticOp::CLAMP: f("lc_clamp"); break;
        case xir::ArithmeticOp::SATURATE: f("lc_saturate"); break;
        case xir::ArithmeticOp::LERP: f("lc_lerp"); break;
        case xir::ArithmeticOp::SMOOTHSTEP: f("lc_smoothstep"); break;
        case xir::ArithmeticOp::STEP: f("lc_step"); break;
        case xir::ArithmeticOp::ABS: f("lc_abs"); break;
        case xir::ArithmeticOp::MIN: f("lc_min"); break;
        case xir::ArithmeticOp::MAX: f("lc_max"); break;
        case xir::ArithmeticOp::CLZ: f("lc_clz"); break;
        case xir::ArithmeticOp::CTZ: f("lc_ctz"); break;
        case xir::ArithmeticOp::POPCOUNT: f("lc_popcount"); break;
        case xir::ArithmeticOp::REVERSE: f("lc_reverse"); break;
        case xir::ArithmeticOp::ISINF: f("lc_isinf"); break;
        case xir::ArithmeticOp::ISNAN: f("lc_isnan"); break;
        case xir::ArithmeticOp::ACOS: f("lc_acos"); break;
        case xir::ArithmeticOp::ACOSH: f("lc_acosh"); break;
        case xir::ArithmeticOp::ASIN: f("lc_asin"); break;
        case xir::ArithmeticOp::ASINH: f("lc_asinh"); break;
        case xir::ArithmeticOp::ATAN: f("lc_atan"); break;
        case xir::ArithmeticOp::ATAN2: f("lc_atan2"); break;
        case xir::ArithmeticOp::ATANH: f("lc_atanh"); break;
        case xir::ArithmeticOp::COS: f("lc_cos"); break;
        case xir::ArithmeticOp::COSH: f("lc_cosh"); break;
        case xir::ArithmeticOp::SIN: f("lc_sin"); break;
        case xir::ArithmeticOp::SINH: f("lc_sinh"); break;
        case xir::ArithmeticOp::TAN: f("lc_tan"); break;
        case xir::ArithmeticOp::TANH: f("lc_tanh"); break;
        case xir::ArithmeticOp::EXP: f("lc_exp"); break;
        case xir::ArithmeticOp::EXP2: f("lc_exp2"); break;
        case xir::ArithmeticOp::EXP10: f("lc_exp10"); break;
        case xir::ArithmeticOp::LOG: f("lc_log"); break;
        case xir::ArithmeticOp::LOG2: f("lc_log2"); break;
        case xir::ArithmeticOp::LOG10: f("lc_log10"); break;
        case xir::ArithmeticOp::POW: f("lc_pow"); break;
        case xir::ArithmeticOp::POW_INT: f("lc_powi"); break;
        case xir::ArithmeticOp::SQRT: f("lc_sqrt"); break;
        case xir::ArithmeticOp::RSQRT: f("lc_rsqrt"); break;
        case xir::ArithmeticOp::CEIL: f("lc_ceil"); break;
        case xir::ArithmeticOp::FLOOR: f("lc_floor"); break;
        case xir::ArithmeticOp::FRACT: f("lc_fract"); break;
        case xir::ArithmeticOp::TRUNC: f("lc_trunc"); break;
        case xir::ArithmeticOp::ROUND: f("lc_round"); break;
        case xir::ArithmeticOp::RINT: f("lc_round"); break;// TODO: check if this is correct
        case xir::ArithmeticOp::FMA: f("lc_fma"); break;
        case xir::ArithmeticOp::COPYSIGN: f("lc_copysign"); break;
        case xir::ArithmeticOp::CROSS: f("lc_cross"); break;
        case xir::ArithmeticOp::DOT: f("lc_dot"); break;
        case xir::ArithmeticOp::LENGTH: f("lc_length"); break;
        case xir::ArithmeticOp::LENGTH_SQUARED: f("lc_length_squared"); break;
        case xir::ArithmeticOp::NORMALIZE: f("lc_normalize"); break;
        case xir::ArithmeticOp::FACEFORWARD: f("lc_faceforward"); break;
        case xir::ArithmeticOp::REFLECT: f("lc_reflect"); break;
        case xir::ArithmeticOp::REDUCE_SUM: f("lc_reduce_sum"); break;
        case xir::ArithmeticOp::REDUCE_PRODUCT: f("lc_reduce_prod"); break;
        case xir::ArithmeticOp::REDUCE_MIN: f("lc_reduce_min"); break;
        case xir::ArithmeticOp::REDUCE_MAX: f("lc_reduce_max"); break;
        case xir::ArithmeticOp::OUTER_PRODUCT: f("lc_outer_product"); break;
        case xir::ArithmeticOp::MATRIX_COMP_NEG: u("-"); break;
        case xir::ArithmeticOp::MATRIX_COMP_ADD: b("+"); break;
        case xir::ArithmeticOp::MATRIX_COMP_SUB: b("-"); break;
        case xir::ArithmeticOp::MATRIX_COMP_MUL: f("lc_mat_comp_mul"); break;
        case xir::ArithmeticOp::MATRIX_COMP_DIV: {
            if (inst->operand(0)->type()->is_scalar() || inst->operand(1)->type()->is_scalar()) {
                b("/");
            } else {
                _emit_with_template(inst, "lc_mat_comp_mul(", 0, ", 1.f / ", 1, ")");
            }
            break;
        }
        case xir::ArithmeticOp::MATRIX_LINALG_MUL: b("*"); break;
        case xir::ArithmeticOp::MATRIX_DETERMINANT: f("lc_determinant"); break;
        case xir::ArithmeticOp::MATRIX_TRANSPOSE: f("lc_transpose"); break;
        case xir::ArithmeticOp::MATRIX_INVERSE: f("lc_inverse"); break;
        case xir::ArithmeticOp::AGGREGATE: break;
        case xir::ArithmeticOp::SHUFFLE: break;
        case xir::ArithmeticOp::INSERT: break;
        case xir::ArithmeticOp::EXTRACT: break;
    }
}

void CUDACodegenXIR::_emit_thread_group_inst(const xir::ThreadGroupInst *inst) noexcept {
    auto f = [&](auto s) noexcept { _emit_intrinsic_call(s, inst); };
    switch (inst->op()) {
        case xir::ThreadGroupOp::SHADER_EXECUTION_REORDER: f("lc_shader_execution_reorder"); break;
        case xir::ThreadGroupOp::RASTER_QUAD_DDX: LUISA_NOT_IMPLEMENTED("lc_raster_quad_ddx"); break;
        case xir::ThreadGroupOp::RASTER_QUAD_DDY: LUISA_NOT_IMPLEMENTED("lc_raster_quad_ddy"); break;
        case xir::ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE: f("lc_warp_is_first_active_lane"); break;
        case xir::ThreadGroupOp::WARP_FIRST_ACTIVE_LANE: f("lc_warp_first_active_lane"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL: f("lc_warp_active_all_equal"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_AND: f("lc_warp_active_bit_and"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_OR: f("lc_warp_active_bit_or"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_XOR: f("lc_warp_active_bit_xor"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_COUNT_BITS: f("lc_warp_active_count_bits"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_MAX: f("lc_warp_active_max"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_MIN: f("lc_warp_active_min"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_PRODUCT: f("lc_warp_active_product"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_SUM: f("lc_warp_active_sum"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_ALL: f("lc_warp_active_all"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_ANY: f("lc_warp_active_any"); break;
        case xir::ThreadGroupOp::WARP_ACTIVE_BIT_MASK: f("lc_warp_active_bit_mask"); break;
        case xir::ThreadGroupOp::WARP_PREFIX_COUNT_BITS: f("lc_warp_prefix_count_bits"); break;
        case xir::ThreadGroupOp::WARP_PREFIX_SUM: f("lc_warp_prefix_sum"); break;
        case xir::ThreadGroupOp::WARP_PREFIX_PRODUCT: f("lc_warp_prefix_product"); break;
        case xir::ThreadGroupOp::WARP_READ_LANE: f("lc_warp_read_lane"); break;
        case xir::ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE: f("lc_warp_read_first_active_lane"); break;
        case xir::ThreadGroupOp::SYNCHRONIZE_BLOCK: f("lc_synchronize_block"); break;
    }
}

void CUDACodegenXIR::_emit_resource_query_inst(const xir::ResourceQueryInst *inst) noexcept {
    switch (inst->op()) {
        case xir::ResourceQueryOp::BUFFER_SIZE: break;
        case xir::ResourceQueryOp::BYTE_BUFFER_SIZE: break;
        case xir::ResourceQueryOp::TEXTURE2D_SIZE: break;
        case xir::ResourceQueryOp::TEXTURE3D_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_BUFFER_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_BYTE_BUFFER_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: break;
        case xir::ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: break;
        case xir::ResourceQueryOp::BUFFER_DEVICE_ADDRESS: break;
        case xir::ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: break;
        case xir::ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: break;
        case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: break;
        case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: break;
    }
}

void CUDACodegenXIR::_emit_resource_read_inst(const xir::ResourceReadInst *inst) noexcept {
}

void CUDACodegenXIR::_emit_resource_write_inst(const xir::ResourceWriteInst *inst) noexcept {
}

void CUDACodegenXIR::_emit_ray_query_object_read_inst(const xir::RayQueryObjectReadInst *inst) noexcept {
}

void CUDACodegenXIR::_emit_ray_query_object_write_inst(const xir::RayQueryObjectWriteInst *inst) noexcept {
}

void CUDACodegenXIR::_emit_branch_inst(const xir::BranchInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(!_control_flow_stack.empty(), "Control flow stack is empty.");
    switch (auto control_flow = _control_flow_stack.back(); control_flow->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::IF: {
            [[maybe_unused]] auto if_inst = static_cast<const xir::IfInst *>(control_flow);
            LUISA_ASSERT(inst->target_block() == if_inst->merge_block(),
                         "Branch target block is not the merge block of the if instruction.");
            break;
        }
        case xir::DerivedInstructionTag::SWITCH: {
            [[maybe_unused]] auto switch_inst = static_cast<const xir::SwitchInst *>(control_flow);
            LUISA_ASSERT(inst->target_block() == switch_inst->merge_block(),
                         "Branch target block is not the merge block of the switch instruction.");
            _scratch << "break;";
            break;
        }
        case xir::DerivedInstructionTag::SIMPLE_LOOP: {
            auto loop = static_cast<const xir::SimpleLoopInst *>(control_flow);
            if (inst->target_block() == loop->merge_block()) {
                _scratch << "break; /* simple loop unconditional branch */";
            }
            break;
        }
        case xir::DerivedInstructionTag::LOOP: {
            auto loop = static_cast<const xir::LoopInst *>(control_flow);
            if (inst->target_block() == loop->merge_block()) {
                _scratch << "break; /* generic loop unconditional branch */";
            }
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION("Control flow stack is not a loop instruction.");
    }
}

void CUDACodegenXIR::_emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst) noexcept {
    LUISA_DEBUG_ASSERT(!_control_flow_stack.empty(), "Control flow stack is empty.");
    switch (auto control_flow = _control_flow_stack.back(); control_flow->derived_instruction_tag()) {
        case xir::DerivedInstructionTag::LOOP: [[fallthrough]];
        case xir::DerivedInstructionTag::SIMPLE_LOOP: {
            if (auto merge_block = control_flow->control_flow_merge()->merge_block()) {
                if (inst->true_block() == merge_block) {// break if true
                    _scratch << "if (";
                    _emit_value_name(inst->condition());
                    _scratch << ") /* loop conditional branch */ { break; }";
                } else if (inst->false_block() == merge_block) {// break if not true
                    _scratch << "if (!(";
                    _emit_value_name(inst->condition());
                    _scratch << ")) /* loop conditional branch */ { break; }";
                }
            }
            break;
        }
        default: LUISA_ERROR_WITH_LOCATION("Control flow stack is not a loop instruction.");
    }
}

void CUDACodegenXIR::_emit_function_definition(const xir::FunctionDefinition *def) noexcept {
    _local_value_indices.clear();
    switch (def->derived_function_tag()) {
        case xir::DerivedFunctionTag::KERNEL: {
            auto kernel = static_cast<const xir::KernelFunction *>(def);
            _emit_kernel_definition(kernel);
            break;
        }
        case xir::DerivedFunctionTag::CALLABLE: {
            auto callable = static_cast<const xir::CallableFunction *>(def);
            _emit_callable_definition(callable);
            break;
        }
        default: LUISA_NOT_IMPLEMENTED(
            "Unsupported function definition {} in XIR-based CUDA codegen.",
            def->name().value_or("unknown"));
    }
}

void CUDACodegenXIR::_emit_kernel_definition(const xir::KernelFunction *kernel) noexcept {
    // declare the kernel argument struct
    _scratch << "struct alignas(16) Params {";
    for (auto arg : kernel->arguments()) {
        LUISA_ASSERT(!arg->is_reference(), "Reference argument is not supported.");
        _scratch << "\n  alignas(16) ";
        _emit_type_name(arg->type());
        _scratch << " ";
        _emit_value_name(arg);
        _scratch << "{};";
    }
    if (_requires_printing) {
        _scratch << "\n  alignas(16) LCPrintBuffer print_buffer{};";
    }
    _scratch << "\n  alignas(16) lc_uint4 ls_kid;";
    _scratch << "\n};\n\n";
    // emit the kernel signature
    if (_requires_optix) {
        _scratch << "extern \"C\" { __constant__ Params params; }\n\n"
                 << "extern \"C\" __global__ void __raygen__main() {";
    } else {
        _scratch << "extern \"C\" __global__ void kernel_main(const Params params) {";
    }
    // decode the kernel arguments
    _scratch << "\n\n  /* kernel arguments */";
    for (auto arg : kernel->arguments()) {
        _scratch << "\n  auto const ";
        _emit_value_name(arg);
        _scratch << " = params.";
        _emit_value_name(arg);
        _scratch << ";";
    }
    if (!_requires_optix && _requires_printing) {
        _scratch << "\n  auto const print_buffer = params.print_buffer;";
    }
    // emit built-in variables
    _scratch
        << "\n\n  /* built-in variables */"
        // block size
        << "\n  constexpr auto bs = lc_block_size();"
        // launch size
        << "\n  const auto ls = lc_dispatch_size();"
        // dispatch id
        << "\n  const auto did = lc_dispatch_id();"
        // thread id
        << "\n  const auto tid = lc_thread_id();"
        // block id
        << "\n  const auto bid = lc_block_id();"
        // kernel id
        << "\n  const auto kid = lc_kernel_id();"
        // warp size
        << "\n  const auto ws = lc_warp_size();"
        // warp lane id
        << "\n  const auto lid = lc_warp_lane_id();";
    // emit launch size check if not using OptiX (Optix handles this internally)
    if (!_requires_optix) {
        _scratch << "\n  if (lc_any(did >= ls)) { return; }";
    }
    _scratch << "\n\n  /* function body */\n";
    _emit_instructions(kernel->body_block()->instructions(), 1);
    _scratch << "}\n\n";
}

void CUDACodegenXIR::_emit_callable_definition(const xir::CallableFunction *callable) noexcept {
    _scratch << "__device__ ";
    _emit_type_name(callable->type());
    _scratch << " ";
    _emit_value_name(callable);
    _scratch << "(";
    auto any_arg = false;
    for (auto arg : callable->arguments()) {
        any_arg = true;
        _scratch << "\n    ";
        _emit_type_name(arg->type());
        if (arg->is_reference()) {
            _scratch << " *const ";
        } else {
            _scratch << " const ";
        }
        _emit_value_name(arg);
        _scratch << ",";
    }
    if (!_requires_optix && _requires_printing) {
        any_arg = true;
        _scratch << "\n    LCPrintBuffer const print_buffer,";
    }
    if (any_arg) { _scratch.pop_back(); }
    _scratch << ") noexcept {\n"
             << "  /* function body */\n";
    _emit_instructions(callable->body_block()->instructions(), 1);
    _scratch << "}\n\n";
}

void CUDACodegenXIR::emit(const xir::Module *module,
                          luisa::string_view device_lib,
                          luisa::string_view native_include) noexcept {

    auto requires_raytracing_closest = false;
    auto requires_raytracing_any = false;
    auto requires_raytracing_query = false;
    auto required_curve_bases = CurveBasisSet::make_none();

    luisa::vector<const xir::FunctionDefinition *> functions_post_order;

    // find the kernel function
    const xir::KernelFunction *kernel = nullptr;
    {
        size_t function_count = 0u;
        for (auto &&f : module->function_list()) {
            function_count++;
            if (f.isa<xir::KernelFunction>()) {
                LUISA_ASSERT(kernel == nullptr,
                             "CUDA codegen: expected exactly one kernel function, "
                             "found {:?}.",
                             f.name().value_or("unknown"));
                kernel = static_cast<const xir::KernelFunction *>(&f);
            }
        }
        LUISA_ASSERT(kernel != nullptr,
                     "CUDA codegen: expected exactly one kernel function, "
                     "found {}.",
                     function_count);
        functions_post_order.reserve(function_count);
    }

    // recursively traverse instructions in the functions to collect the basic information
    luisa::unordered_set<const Type *> types;
    luisa::unordered_set<const xir::Constant *> constants;
    types.reserve(Type::count());
    constants.reserve(64u);
    {
        luisa::unordered_set<const xir::Function *> visited;
        visited.reserve(functions_post_order.capacity());
        auto traverse = [&](auto &&self, const xir::Function *f) noexcept {
            if (!visited.emplace(f).second) { return; }
            if (auto def = f->definition()) {
                def->traverse_instructions([&](const xir::Instruction *inst) noexcept {
                    if (inst->isa<xir::PrintInst>()) {
                        this->_requires_printing = true;
                        auto print = static_cast<const xir::PrintInst *>(inst);
                        auto [iter, success] = _print_info.emplace(print, PrintInfo{nullptr, _print_formats.size()});
                        LUISA_ASSERT(success, "Print info already exists.");
                        luisa::vector<const Type *> arg_types;
                        arg_types.reserve(print->operand_count() + 2u);
                        arg_types.emplace_back(Type::of<uint>());// arg size
                        arg_types.emplace_back(Type::of<uint>());// fmt id
                        for (auto op_use : print->operand_uses()) {
                            LUISA_ASSERT(op_use->value() != nullptr, "Print operand use is null.");
                            arg_types.emplace_back(op_use->value()->type());
                        }
                        auto s = Type::structure(arg_types);
                        types.emplace(s);
                        iter->second.type = s;
                        _print_formats.emplace_back(print->format(), s);
                    } else if (inst->isa<xir::ResourceQueryInst>()) {
                        auto is_ray_trace = false;
                        switch (static_cast<const xir::ResourceQueryInst *>(inst)->op()) {
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: {
                                this->_requires_optix = true;
                                requires_raytracing_closest = true;
                                is_ray_trace = true;
                                break;
                            }
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
                                this->_requires_optix = true;
                                requires_raytracing_any = true;
                                is_ray_trace = true;
                                break;
                            }
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
                                this->_requires_optix = true;
                                requires_raytracing_query = true;
                                is_ray_trace = true;
                                break;
                            }
                            default: break;
                        }
                        if (is_ray_trace) {
                            if (auto curve_basis_md = inst->find_metadata<xir::CurveBasisMD>()) {
                                required_curve_bases.propagate(curve_basis_md->curve_basis_set());
                            }
                        }
                    }
                    types.emplace(inst->type());
                    for (auto op_use : inst->operand_uses()) {
                        if (auto value = op_use->value()) {
                            types.emplace(value->type());
                            if (value->isa<xir::Constant>()) {
                                constants.emplace(static_cast<const xir::Constant *>(value));
                            } else if (value->isa<xir::Function>()) {
                                self(self, static_cast<const xir::Function *>(value));
                            }
                        }
                    }
                });
                functions_post_order.emplace_back(def);
            }
        };
        traverse(traverse, kernel);
    }
    LUISA_ASSERT(!functions_post_order.empty() && functions_post_order.back() == kernel,
                 "CUDA codegen: kernel function not found in post order traversal.");

    // generate macro definitions and header
    if (_requires_optix) {
        _scratch << "#define LUISA_ENABLE_OPTIX\n";
        if (required_curve_bases.any()) {
            _scratch << "#define LUISA_ENABLE_OPTIX_CURVE\n";
        }
        if (requires_raytracing_closest) {
            _scratch << "#define LUISA_ENABLE_OPTIX_TRACE_CLOSEST\n";
        }
        if (requires_raytracing_any) {
            _scratch << "#define LUISA_ENABLE_OPTIX_TRACE_ANY\n";
        }
        if (requires_raytracing_query) {
            _scratch << "#define LUISA_ENABLE_OPTIX_RAY_QUERY\n";
        }
    }
    _scratch << "#define LC_BLOCK_SIZE lc_make_uint3("
             << kernel->block_size().x << ", "
             << kernel->block_size().y << ", "
             << kernel->block_size().z << ")\n"
             << "\n/* built-in device library begin */\n"
             << device_lib
             << "\n/* built-in device library end */\n\n";

    // emit type declarations
    _emit_type_definitions(std::move(types));

    // emit global constants
    _emit_global_constants(std::move(constants));

    // emit native include if any
    if (!native_include.empty()) {
        _scratch << "\n/* native include begin */\n\n"
                 << native_include
                 << "\n/* native include end */\n\n";
    }

    // emit function definitions
    for (auto f : functions_post_order) {
        _emit_function_definition(f);
    }

    // emit indirect dispatch kernel if allowed
    if (_allow_indirect_dispatch && !_requires_optix) {
        _scratch << "extern \"C\" __global__ void kernel_launcher(Params params, const LCIndirectBuffer indirect) {\n"
                 << "  auto i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                 << "  auto n = min(indirect.header()->size, indirect.capacity - indirect.offset);\n"
                 << "  if (i < n) {\n"
                 << "    auto args = params;\n"
                 << "    auto d = indirect.dispatches()[i + indirect.offset];\n"
                 << "    args.ls_kid = d.dispatch_size_and_kernel_id;\n"
                 << "    auto block_size = lc_block_size();\n"
                 << "#ifdef LUISA_DEBUG\n"
                 << "    lc_assert(lc_all(block_size == d.block_size));\n"
                 << "#endif\n"
                 << "    auto dispatch_size = lc_make_uint3(d.dispatch_size_and_kernel_id);\n"
                 << "    if (lc_all(dispatch_size > 0u)) {\n"
                 << "      auto block_count = (dispatch_size + block_size - 1u) / block_size;\n"
                 << "      auto nb = dim3(block_count.x, block_count.y, block_count.z);\n"
                 << "      auto bs = dim3(block_size.x, block_size.y, block_size.z);\n"
                 << "      kernel_main<<<nb, bs>>>(args);\n"
                 << "    }\n"
                 << "  }\n"
                 << "}\n\n";
    }
}

}// namespace luisa::compute::cuda

#endif
