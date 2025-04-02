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

public:
    CUDACodegenXIR(StringScratch &scratch, bool allow_indirect) noexcept;
    ~CUDACodegenXIR() noexcept;
    void emit(const xir::Module *module, luisa::string_view device_lib, luisa::string_view native_include) noexcept;
    [[nodiscard]] auto move_print_formats() && noexcept { return std::move(_print_formats); }
};

}// namespace luisa::compute::cuda

#endif
