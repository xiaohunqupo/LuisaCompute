#include <luisa/core/logging.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/ray_query.h>

namespace luisa::compute::xir {

RayQueryObjectReadInst::RayQueryObjectReadInst(const Type *type, RayQueryObjectReadOp op,
                                               luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{type},
      InstructionOpMixin{op} { set_operands(operands); }

RayQueryObjectWriteInst::RayQueryObjectWriteInst(RayQueryObjectWriteOp op,
                                                 luisa::span<Value *const> operands) noexcept
    : DerivedInstruction{nullptr},
      InstructionOpMixin{op} { set_operands(operands); }

RayQueryLoopInst::RayQueryLoopInst() noexcept {
    auto dispatch_block = static_cast<Value *>(nullptr);
    auto operands = std::array{dispatch_block};
    set_operands(operands);
}

void RayQueryLoopInst::set_dispatch_block(BasicBlock *block) noexcept {
    set_operand(operand_index_dispatch_block, block);
}

BasicBlock *RayQueryLoopInst::create_dispatch_block() noexcept {
    auto block = Pool::current()->create<BasicBlock>();
    set_dispatch_block(block);
    return block;
}

BasicBlock *RayQueryLoopInst::dispatch_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_dispatch_block));
}

const BasicBlock *RayQueryLoopInst::dispatch_block() const noexcept {
    return const_cast<RayQueryLoopInst *>(this)->dispatch_block();
}

RayQueryDispatchInst::RayQueryDispatchInst(Value *query_object) noexcept {
    auto exit_block = static_cast<Value *>(nullptr);
    auto on_surface_candidate_block = static_cast<Value *>(nullptr);
    auto on_procedural_candidate_block = static_cast<Value *>(nullptr);
    auto operands = std::array{query_object, exit_block, on_surface_candidate_block, on_procedural_candidate_block};
    LUISA_DEBUG_ASSERT(operands[operand_index_query_object] == query_object, "Invalid query object operand.");
    set_operands(operands);
}

void RayQueryDispatchInst::set_query_object(Value *query_object) noexcept {
    set_operand(operand_index_query_object, query_object);
}

void RayQueryDispatchInst::set_exit_block(BasicBlock *block) noexcept {
    set_operand(operand_index_exit_block, block);
}

void RayQueryDispatchInst::set_on_surface_candidate_block(BasicBlock *block) noexcept {
    set_operand(operand_index_on_surface_candidate_block, block);
}

void RayQueryDispatchInst::set_on_procedural_candidate_block(BasicBlock *block) noexcept {
    set_operand(operand_index_on_procedural_candidate_block, block);
}

BasicBlock *RayQueryDispatchInst::create_on_surface_candidate_block() noexcept {
    auto block = Pool::current()->create<BasicBlock>();
    set_on_surface_candidate_block(block);
    return block;
}

BasicBlock *RayQueryDispatchInst::create_on_procedural_candidate_block() noexcept {
    auto block = Pool::current()->create<BasicBlock>();
    set_on_procedural_candidate_block(block);
    return block;
}

Value *RayQueryDispatchInst::query_object() noexcept {
    return operand(operand_index_query_object);
}

const Value *RayQueryDispatchInst::query_object() const noexcept {
    return operand(operand_index_query_object);
}

BasicBlock *RayQueryDispatchInst::exit_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_exit_block));
}

const BasicBlock *RayQueryDispatchInst::exit_block() const noexcept {
    return const_cast<RayQueryDispatchInst *>(this)->exit_block();
}

BasicBlock *RayQueryDispatchInst::on_surface_candidate_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_on_surface_candidate_block));
}

const BasicBlock *RayQueryDispatchInst::on_surface_candidate_block() const noexcept {
    return const_cast<RayQueryDispatchInst *>(this)->on_surface_candidate_block();
}

BasicBlock *RayQueryDispatchInst::on_procedural_candidate_block() noexcept {
    return static_cast<BasicBlock *>(operand(operand_index_on_procedural_candidate_block));
}

const BasicBlock *RayQueryDispatchInst::on_procedural_candidate_block() const noexcept {
    return const_cast<RayQueryDispatchInst *>(this)->on_procedural_candidate_block();
}

RayQueryPipelineInst::RayQueryPipelineInst(Value *query_object, Value *query_context,
                                           Function *on_surface, Function *on_procedural) noexcept {
    std::array operands{query_object, query_context, static_cast<Value *>(on_surface), static_cast<Value *>(on_procedural)};
    LUISA_DEBUG_ASSERT(operands[operand_index_query_object] == query_object, "Invalid query object operand.");
    LUISA_DEBUG_ASSERT(operands[operand_index_query_context] == query_context, "Invalid query context operand.");
    LUISA_DEBUG_ASSERT(operands[operand_index_on_surface_function] == on_surface, "Invalid on surface function operand.");
    LUISA_DEBUG_ASSERT(operands[operand_index_on_procedural_function] == on_procedural, "Invalid on procedural function operand.");
    set_operands(operands);
}

void RayQueryPipelineInst::set_query_object(Value *query_object) noexcept {
    set_operand(operand_index_query_object, query_object);
}

void RayQueryPipelineInst::set_query_context(Value *query_context) noexcept {
    set_operand(operand_index_query_context, query_context);
}

void RayQueryPipelineInst::set_on_surface_function(Function *on_surface) noexcept {
    set_operand(operand_index_on_surface_function, on_surface);
}

void RayQueryPipelineInst::set_on_procedural_function(Function *on_procedural) noexcept {
    set_operand(operand_index_on_procedural_function, on_procedural);
}

Value *RayQueryPipelineInst::query_object() noexcept {
    return operand(operand_index_query_object);
}

const Value *RayQueryPipelineInst::query_object() const noexcept {
    return operand(operand_index_query_object);
}

Value *RayQueryPipelineInst::query_context() noexcept {
    return operand(operand_index_query_context);
}

const Value *RayQueryPipelineInst::query_context() const noexcept {
    return operand(operand_index_query_context);
}

Function *RayQueryPipelineInst::on_surface_function() noexcept {
    auto func = operand(operand_index_on_surface_function);
    LUISA_DEBUG_ASSERT(func->derived_value_tag() == DerivedValueTag::FUNCTION, "Invalid on surface function operand.");
    return static_cast<Function *>(func);
}

const Function *RayQueryPipelineInst::on_surface_function() const noexcept {
    return const_cast<RayQueryPipelineInst *>(this)->on_surface_function();
}

Function *RayQueryPipelineInst::on_procedural_function() noexcept {
    auto func = operand(operand_index_on_procedural_function);
    LUISA_DEBUG_ASSERT(func->derived_value_tag() == DerivedValueTag::FUNCTION, "Invalid on procedural function operand.");
    return static_cast<Function *>(func);
}

const Function *RayQueryPipelineInst::on_procedural_function() const noexcept {
    return const_cast<RayQueryPipelineInst *>(this)->on_procedural_function();
}

}// namespace luisa::compute::xir
