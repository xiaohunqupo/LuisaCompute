#pragma once

#ifdef LUISA_ENABLE_XIR

#include <luisa/core/string_scratch.h>
#include <luisa/xir/builder.h>

namespace luisa::compute::cuda {

class CUDACodegenXIR {

private:
    StringScratch &_scratch;
    luisa::unordered_map<const xir::PrintInst *, const Type *> _print_stmt_types;
    luisa::vector<std::pair<luisa::string, const Type *>> _print_formats;
    luisa::unordered_map<luisa::string, uint> _string_ids;
    uint32_t _indent{0u};
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

public:
    CUDACodegenXIR(StringScratch &scratch, bool allow_indirect) noexcept;
    ~CUDACodegenXIR() noexcept;
    void emit(const xir::Module *module,
              luisa::string_view device_lib,
              luisa::string_view native_include) noexcept;
    [[nodiscard]] auto print_formats() const noexcept {
        return luisa::span{_print_formats};
    }
};

}// namespace luisa::compute::cuda

#endif
