#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/ast/type.h>
#include <luisa/xir/value.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/debug_printer.h>

namespace luisa::compute::xir {

using namespace std::string_view_literals;

struct XIRDebugPrinter::Impl {

    luisa::unordered_map<const Value *, size_t> value_indices;

    void reset() noexcept {
        value_indices.clear();
    }
};

XIRDebugPrinter::XIRDebugPrinter() noexcept
    : _impl{luisa::make_unique<Impl>()} {}

void XIRDebugPrinter::reset() noexcept {
    _impl->reset();
}

size_t XIRDebugPrinter::value_uid(const Value *value) noexcept {
    auto it = _impl->value_indices.try_emplace(value, _impl->value_indices.size());
    return it.first->second;
}

void XIRDebugPrinter::emit_type(luisa::string &s, const Type *type) noexcept {
    if (type == nullptr) {
        s.append("void"sv);
        return;
    }
    switch (type->tag()) {
        case Type::Tag::BOOL: s.append("bool"sv); break;
        case Type::Tag::INT8: s.append("i8"sv); break;
        case Type::Tag::UINT8: s.append("u8"sv); break;
        case Type::Tag::INT16: s.append("i16"sv); break;
        case Type::Tag::UINT16: s.append("u16"sv); break;
        case Type::Tag::INT32: s.append("i32"sv); break;
        case Type::Tag::UINT32: s.append("u32"sv); break;
        case Type::Tag::INT64: s.append("i64"sv); break;
        case Type::Tag::UINT64: s.append("u64"sv); break;
        case Type::Tag::FLOAT16: s.append("f16"sv); break;
        case Type::Tag::FLOAT32: s.append("f32"sv); break;
        case Type::Tag::FLOAT64: s.append("f64"sv); break;
        case Type::Tag::VECTOR: {
            s.append("<"sv);
            emit_type(s, type->element());
            luisa::format_to(std::back_inserter(s), " x {}>", type->dimension());
            break;
        }
        case Type::Tag::MATRIX: {
            s.append("("sv);
            emit_type(s, type->element());
            auto n = type->dimension();
            luisa::format_to(std::back_inserter(s), " x {} x {})", n, n);
            break;
        }
        case Type::Tag::ARRAY: {
            s.append("["sv);
            emit_type(s, type->element());
            luisa::format_to(std::back_inserter(s), " x {}]", type->dimension());
            break;
        }
        case Type::Tag::STRUCTURE: {
            luisa::format_to(std::back_inserter(s), "{{{}", type->alignment());
            for (auto m : type->members()) {
                s.append(", "sv);
                emit_type(s, m);
            }
            s.append("}"sv);
            break;
        }
        case Type::Tag::BUFFER: {
            s.append("Buffer<"sv);
            emit_type(s, type->element());
            s.append(">"sv);
            break;
        }
        case Type::Tag::TEXTURE: {
            luisa::format_to(std::back_inserter(s), "Texture{}D<", type->element()->dimension());
            emit_type(s, type->element());
            s.append(">"sv);
            break;
        }
        case Type::Tag::BINDLESS_ARRAY: s.append("Bindless"sv); break;
        case Type::Tag::ACCEL: s.append("Accel"sv); break;
        case Type::Tag::CUSTOM: s.append(R"(")").append(type->description()).append(R"(")"); break;
        default: LUISA_ERROR_WITH_LOCATION("Unknown type");
    }
}

void XIRDebugPrinter::emit_value_name(luisa::string &s, const Value *value) noexcept {
    if (value == nullptr) {
        s.append("null"sv);
        return;
    }
    if (value->isa<SpecialRegister>()) {
        auto sreg = static_cast<const SpecialRegister *>(value);
        luisa::format_to(std::back_inserter(s), "%{}",
                         xir::to_string(sreg->derived_special_register_tag()));
        return;
    }
    auto uid = value_uid(value);
    luisa::format_to(std::back_inserter(s), "%{}", uid);
}

void XIRDebugPrinter::emit_operand(luisa::string &s, const Value *value) noexcept {
    if (value == nullptr) {
        return emit_value_name(s, value);
    }
    if (!value->isa<BasicBlock>() && !value->isa<Function>()) {
        emit_type(s, value->type());
        s.append(" "sv);
    }
    if (value->isa<SpecialRegister>()) {
        return emit_value_name(s, value);
    }
    luisa::format_to(std::back_inserter(s), " {} ",
                     xir::to_string(value->derived_value_tag()));
    emit_value_name(s, value);
}

void XIRDebugPrinter::emit_instruction(luisa::string &s, const Instruction *instruction, int indent) noexcept {
    LUISA_DEBUG_ASSERT(instruction != nullptr);
    s.append(2 * indent, ' ');
    emit_type(s, instruction->type());
    s.append(" "sv);
    emit_value_name(s, instruction);
    luisa::format_to(std::back_inserter(s), " = {}", instruction->intrinsic_identifier());
    switch (instruction->derived_instruction_tag()) {
        case DerivedInstructionTag::IF: break;
        case DerivedInstructionTag::SWITCH: break;
        case DerivedInstructionTag::LOOP: break;
        case DerivedInstructionTag::SIMPLE_LOOP: break;
        case DerivedInstructionTag::BRANCH: break;
        case DerivedInstructionTag::CONDITIONAL_BRANCH: break;
        case DerivedInstructionTag::UNREACHABLE: break;
        case DerivedInstructionTag::BREAK: break;
        case DerivedInstructionTag::CONTINUE: break;
        case DerivedInstructionTag::RETURN: break;
        case DerivedInstructionTag::RASTER_DISCARD: break;
        case DerivedInstructionTag::PHI: break;
        case DerivedInstructionTag::ALLOCA: break;
        case DerivedInstructionTag::LOAD: break;
        case DerivedInstructionTag::STORE: break;
        case DerivedInstructionTag::GEP: break;
        case DerivedInstructionTag::ATOMIC: break;
        case DerivedInstructionTag::ARITHMETIC: break;
        case DerivedInstructionTag::THREAD_GROUP: break;
        case DerivedInstructionTag::RESOURCE_QUERY: break;
        case DerivedInstructionTag::RESOURCE_READ: break;
        case DerivedInstructionTag::RESOURCE_WRITE: break;
        case DerivedInstructionTag::RAY_QUERY_LOOP: break;
        case DerivedInstructionTag::RAY_QUERY_DISPATCH: break;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: break;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: break;
        case DerivedInstructionTag::RAY_QUERY_PIPELINE: break;
        case DerivedInstructionTag::AUTODIFF_SCOPE: break;
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: break;
        case DerivedInstructionTag::CALL: break;
        case DerivedInstructionTag::CAST: break;
        case DerivedInstructionTag::PRINT: break;
        case DerivedInstructionTag::CLOCK: break;
        case DerivedInstructionTag::ASSERT: break;
        case DerivedInstructionTag::ASSUME: break;
        case DerivedInstructionTag::OUTLINE: break;
    }
}

void XIRDebugPrinter::emit_basic_block(luisa::string &s, const BasicBlock *block, int indent) noexcept {
}

void XIRDebugPrinter::emit_constant(luisa::string &s, const Constant *value) noexcept {
}

void XIRDebugPrinter::emit_function_decl(luisa::string &s, const Function *function) noexcept {
}

void XIRDebugPrinter::emit_function(luisa::string &s, const Function *function) noexcept {
    emit_function_decl(s, function);
    if (auto def = function->definition()) {

    }
}

void XIRDebugPrinter::emit_module(luisa::string &s, const Module *module) noexcept {
    for (auto &c : module->constant_list()) {
        emit_constant(s, &c);
        s.append("\n"sv);
    }
    if (!module->constant_list().empty()) {
        s.append("\n"sv);
    }
    for (auto &f : module->function_list()) {
        emit_function(s, &f);
        s.append("\n\n"sv);
    }
}

}// namespace luisa::compute::xir
