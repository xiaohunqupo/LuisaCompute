//
// Created by mike on 4/1/25.
//

#ifdef LUISA_ENABLE_XIR

#include <luisa/core/stl/algorithm.h>
#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/rtx/ray_query.h>
#include <luisa/xir/module.h>
#include <luisa/xir/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/metadata/curve_basis.h>

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
        case xir::DerivedValueTag::INSTRUCTION: _scratch << "v" << get_local_index(value); break;
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
        case xir::DerivedValueTag::ARGUMENT: _scratch << "a" << get_local_index(value); break;
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
                        switch (static_cast<const xir::ResourceQueryInst *>(inst)->op()) {
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: {
                                this->_requires_optix = true;
                                requires_raytracing_closest = true;
                                break;
                            }
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
                                this->_requires_optix = true;
                                requires_raytracing_any = true;
                                break;
                            }
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
                            case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
                                this->_requires_optix = true;
                                requires_raytracing_query = true;
                                break;
                            }
                            default: break;
                        }
                        if (auto curve_basis_md = inst->find_metadata<xir::CurveBasisMD>()) {
                            required_curve_bases.propagate(curve_basis_md->curve_basis_set());
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
}

}// namespace luisa::compute::cuda

#endif
