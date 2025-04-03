#pragma once

#ifdef LUISA_ENABLE_XIR

#include <luisa/core/string_scratch.h>
#include <luisa/xir/builder.h>

namespace luisa::compute::cuda {

class CUDACodegenXIR {

private:
    struct PrintInfo {
        const Type *type;
        size_t index;
    };
    StringScratch &_scratch;
    luisa::unordered_map<const xir::PrintInst *, PrintInfo> _print_info;
    luisa::vector<std::pair<luisa::string, const Type *>> _print_formats;
    luisa::vector<const xir::Instruction *> _control_flow_stack;
    bool _allow_indirect_dispatch;
    bool _requires_printing{false};
    bool _requires_optix{false};

private:
    const Type *_ray_type;
    const Type *_triangle_hit_type;
    const Type *_procedural_hit_type;
    const Type *_committed_hit_type;
    const Type *_ray_query_all_type;
    const Type *_ray_query_any_type;
    const Type *_indirect_buffer_type;
    const Type *_motion_srt_type;

private:
    luisa::unordered_map<const xir::Value *, size_t> _local_value_indices;
    luisa::unordered_map<const xir::Value *, size_t> _global_value_indices;

private:
    [[nodiscard]] static bool _should_emit_global_constant(const xir::Constant *c) noexcept;
    [[nodiscard]] bool _is_builtin_type(const Type *t) const noexcept;
    void _emit_type_name(const Type *type) noexcept;
    void _emit_type_definition(const Type *type, luisa::unordered_set<const Type *> &defined_types) noexcept;
    void _emit_type_definitions(luisa::unordered_set<const Type *> used_types) noexcept;
    void _emit_value_name(const xir::Value *value, bool is_use = true) noexcept;
    void _emit_global_constants(luisa::unordered_set<const xir::Constant *> used_constants) noexcept;
    void _emit_function_definition(const xir::FunctionDefinition *def) noexcept;
    void _emit_kernel_definition(const xir::KernelFunction *kernel) noexcept;
    void _emit_callable_definition(const xir::CallableFunction *callable) noexcept;
    void _emit_instructions(const xir::InstructionList &inst_list, int indent) noexcept;
    void _emit_metadata(const xir::MetadataList &md_list, int indent) const noexcept;
    void _emit_indent(int indent) const noexcept;
    void _emit_if_inst(const xir::IfInst *inst, int indent) noexcept;
    void _emit_switch_inst(const xir::SwitchInst *inst, int indent) noexcept;
    void _emit_loop_inst(const xir::LoopInst *inst, int indent) noexcept;
    void _emit_simple_loop_inst(const xir::SimpleLoopInst *inst, int indent) noexcept;
    void _emit_gep_inst(const xir::GEPInst *inst, int indent) noexcept;
    void _emit_atomic_inst(const xir::AtomicInst *inst, int indent) noexcept;
    void _emit_arithmetic_inst(const xir::ArithmeticInst *inst, int indent) noexcept;
    void _emit_thread_group_inst(const xir::ThreadGroupInst *inst, int indent) noexcept;
    void _emit_resource_query_inst(const xir::ResourceQueryInst *inst, int indent) noexcept;
    void _emit_resource_read_inst(const xir::ResourceReadInst *inst, int indent) noexcept;
    void _emit_resource_write_inst(const xir::ResourceWriteInst *inst, int indent) noexcept;
    void _emit_ray_query_object_read_inst(const xir::RayQueryObjectReadInst *inst, int indent) noexcept;
    void _emit_ray_query_object_write_inst(const xir::RayQueryObjectWriteInst *inst, int indent) noexcept;
    void _emit_branch_inst(const xir::BranchInst *inst, int indent) noexcept;
    void _emit_conditional_branch_inst(const xir::ConditionalBranchInst *inst, int indent) noexcept;

    template<typename F>
    void _with_control_flow(const xir::Instruction *inst, F &&f) noexcept {
        _control_flow_stack.emplace_back(inst);
        f();
        assert(!_control_flow_stack.empty() && _control_flow_stack.back() == inst && "Control flow stack mismatch.");
        _control_flow_stack.pop_back();
    }

public:
    CUDACodegenXIR(StringScratch &scratch, bool allow_indirect) noexcept;
    ~CUDACodegenXIR() noexcept;
    void emit(const xir::Module *module, luisa::string_view device_lib, luisa::string_view native_include) noexcept;
    [[nodiscard]] auto move_print_formats() && noexcept { return std::move(_print_formats); }
};

}// namespace luisa::compute::cuda

#endif
