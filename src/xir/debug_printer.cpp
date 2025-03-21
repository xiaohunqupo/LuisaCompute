#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/ast/type.h>
#include <luisa/xir/value.h>
#include <luisa/xir/instruction.h>
#include <luisa/xir/function.h>
#include <luisa/xir/module.h>
#include <luisa/xir/metadata/location.h>
#include <luisa/xir/metadata/comment.h>
#include <luisa/xir/debug_printer.h>

namespace luisa::compute::xir {

using namespace std::string_view_literals;

struct XIRDebugPrinter::Impl {

    luisa::unordered_map<const Value *, size_t> value_indices;

    void reset() noexcept {
        value_indices.clear();
    }

    [[nodiscard]] auto index_of(const Value *value) noexcept {
        auto it = value_indices.try_emplace(value, value_indices.size());
        return it.first->second;
    }
};

XIRDebugPrinter::XIRDebugPrinter() noexcept
    : _impl{luisa::make_unique<Impl>()} {}

XIRDebugPrinter::~XIRDebugPrinter() noexcept = default;

void XIRDebugPrinter::reset() noexcept {
    _impl->reset();
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
            luisa::format_to(std::back_inserter(s), "{{align {}", type->alignment());
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
        case Type::Tag::CUSTOM: s.append(R"(T")").append(type->description()).append(R"(")"); break;
        default: LUISA_ERROR_WITH_LOCATION("Unknown type");
    }
}

namespace {

void print_literal(luisa::string &s, const Type *type, const void *data) noexcept {
    s.append("[placeholder]"sv);// TODO
}

}// namespace

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
    if (value->isa<Constant>() && value->type()->is_basic()) {
        return print_literal(s, value->type(), static_cast<const Constant *>(value)->data());
    }
    auto uid = _impl->index_of(value);
    if (value->is_lvalue()) { s.append("*"sv); }
    luisa::format_to(std::back_inserter(s), "%{}", uid);
}

void XIRDebugPrinter::emit_value_debug_info(luisa::string &s, const Value *value) noexcept {
    if (value != nullptr) {
        auto any_info = false;
        constexpr auto prefix = " // "sv;
        s.append(prefix);
        if (!value->metadata_list().empty()) {
            any_info = true;
            s.append("metadata = {"sv);
            for (auto &m : value->metadata_list()) {
                emit_metadata(s, &m);
                s.append(", "sv);
            }
            s.pop_back();
            s.pop_back();
            s.append("}, "sv);
        }
        if (value->isa<Instruction>()) {
            auto inst = static_cast<const Instruction *>(value);
            if (auto merge = inst->control_flow_merge()) {
                any_info = true;
                s.append("merge = "sv);
                emit_value_name(s, merge->merge_block());
                s.append(", "sv);
            }
        }
        if (!value->use_list().empty()) {
            any_info = true;
            s.append("users = {"sv);
            for (auto &use : value->use_list()) {
                emit_value_name(s, use.user());
                s.append(", "sv);
            }
            s.pop_back();
            s.pop_back();
            s.append("}, "sv);
        }
        if (value->isa<BasicBlock>()) {
            auto bb = static_cast<const BasicBlock *>(value);
            auto any_pred = false;
            bb->traverse_predecessors(false, [&](const BasicBlock *pred) noexcept {
                if (any_pred == false) {
                    any_pred = true;
                    s.append("preds = {"sv);
                }
                emit_value_name(s, pred);
                s.append(", "sv);
            });
            if (any_pred) {
                any_info = true;
                s.pop_back();
                s.pop_back();
                s.append("}, "sv);
            }
        }
        if (any_info) {
            s.pop_back();
            s.pop_back();
        } else {
            for (auto _ : prefix) {
                s.pop_back();
            }
        }
    }
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
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(value->derived_value_tag()));
    emit_value_name(s, value);
}

void XIRDebugPrinter::emit_instruction(luisa::string &s, const Instruction *instruction) noexcept {
    LUISA_DEBUG_ASSERT(instruction != nullptr);
    emit_value_name(s, instruction);
    s.append(": "sv);
    if (auto t = instruction->type()) {
        emit_type(s, t);
        s.append(" "sv);
    }
    luisa::format_to(std::back_inserter(s), "{}",
                     instruction->intrinsic_identifier());
    for (auto op_use : instruction->operand_uses()) {
        if (op_use == instruction->operand_uses().front()) {
            s.append(" "sv);
        } else {
            s.append(", "sv);
        }
        emit_operand(s, op_use->value());
    }
    emit_value_debug_info(s, instruction);
}

void XIRDebugPrinter::emit_basic_block(luisa::string &s, const BasicBlock *block) noexcept {
    LUISA_DEBUG_ASSERT(block != nullptr);
    s.append("\n  "sv);
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(block->derived_value_tag()));
    emit_value_name(s, block);
    s.append(": {"sv);
    emit_value_debug_info(s, block);
    s.append("\n"sv);
    for (auto &inst : block->instructions()) {
        s.append("    "sv);
        emit_instruction(s, &inst);
        s.append("\n"sv);
    }
    luisa::format_to(std::back_inserter(s), "  }} // end of {} ",
                     xir::to_string(block->derived_value_tag()));
    emit_value_name(s, block);
}

void XIRDebugPrinter::emit_constant(luisa::string &s, const Constant *value) noexcept {
    LUISA_DEBUG_ASSERT(value != nullptr);
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(Constant::static_derived_value_tag()));
    emit_value_name(s, value);
    s.append(": "sv);
    emit_type(s, value->type());
    s.append(" "sv);
    print_literal(s, value->type(), value->data());
    emit_value_debug_info(s, value);
}

void XIRDebugPrinter::emit_function_decl(luisa::string &s, const Function *function) noexcept {
    LUISA_DEBUG_ASSERT(function != nullptr);
    luisa::format_to(std::back_inserter(s), "{} ",
                     xir::to_string(Function::static_derived_value_tag()));
    emit_value_name(s, function);
    s.append(": "sv);
    emit_type(s, function->type());
    luisa::format_to(std::back_inserter(s), " {}"sv,
                     xir::to_string(function->derived_function_tag()));
    emit_value_debug_info(s, function);
    s.append("\n"sv);
    s.append("("sv);
    for (auto arg : function->arguments()) {
        s.append("\n  "sv);
        emit_value_name(s, arg);
        s.append(": "sv);
        emit_type(s, arg->type());
        emit_value_debug_info(s, arg);
    }
    if (!function->arguments().empty()) {
        s.append("\n"sv);
    }
    s.append(")"sv);
}

void XIRDebugPrinter::emit_function(luisa::string &s, const Function *function) noexcept {
    emit_function_decl(s, function);
    if (auto def = function->definition()) {
        s.append(" {"sv);
        def->traverse_basic_blocks(
            BasicBlockTraversalOrder::REVERSE_POST_ORDER,
            [this, &s](const BasicBlock *block) {
                this->emit_basic_block(s, block);
                s.append("\n"sv);
            });
        s.append("}"sv);
    }
}

void XIRDebugPrinter::emit_module(luisa::string &s, const Module *module) noexcept {
    s.append("module"sv);
    if (auto name = module->name()) {
        luisa::format_to(std::back_inserter(s), "({})", name.value());
    }
    s.append("\n");
    auto any_const = false;
    for (auto &c : module->constant_list()) {
        if (!c.type()->is_basic()) {
            any_const = true;
            s.append("\n"sv);
            emit_constant(s, &c);
        }
    }
    if (any_const) {
        s.append("\n"sv);
    }
    s.append("\n"sv);
    for (auto &f : module->function_list()) {
        emit_function(s, &f);
        s.append("\n\n"sv);
    }
}

void XIRDebugPrinter::emit_metadata(luisa::string &s, const Metadata *metadata) noexcept {
}

}// namespace luisa::compute::xir
