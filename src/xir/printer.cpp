#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/ast/type.h>
#include <luisa/xir/value.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/printer.h>

namespace luisa::compute::xir {

using namespace std::string_view_literals;

struct XIRPrinter::Impl {

    luisa::unordered_map<const Value *, size_t> value_indices;

    void reset() noexcept {
        value_indices.clear();
    }
};

XIRPrinter::XIRPrinter() noexcept
    : _impl{luisa::make_unique<Impl>()} {}

void XIRPrinter::reset() noexcept {
    _impl->reset();
}

size_t XIRPrinter::value_uid(const Value *value) noexcept {
    auto it = _impl->value_indices.try_emplace(value, _impl->value_indices.size());
    return it.first->second;
}

void XIRPrinter::emit_type(luisa::string &s, const Type *type) noexcept {
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

void XIRPrinter::emit_value_name(luisa::string &s, const Value *value) noexcept {
    if (value == nullptr) {
        s.append("null"sv);
        return;
    }
    switch (value->derived_value_tag()) {

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

void XIRPrinter::emit_operand(luisa::string &s, const Value *value, NestedBlockFormat block_format) noexcept {
    if (value == nullptr || value->isa<SpecialRegister>()) {
        emit_value_name(s, value);
        return;
    }

}

void XIRPrinter::emit_instruction(luisa::string &s, const Instruction *instruction, NestedBlockFormat block_format) noexcept {
}

void XIRPrinter::emit_basic_block(luisa::string &s, const BasicBlock *block, NestedBlockFormat block_format) noexcept {
}

void XIRPrinter::emit_constant(luisa::string &s, const Constant *value) noexcept {

}

void XIRPrinter::emit_function_decl(luisa::string &s, const Function *function) noexcept {
}

void XIRPrinter::emit_function(luisa::string &s, const Function *function) noexcept {
}

void XIRPrinter::emit_module(luisa::string &s, const Module *module) noexcept {
    for (auto &sreg : module->special_register_list()) {
    }
}

}// namespace luisa::compute::xir
