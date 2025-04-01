//
// Created by mike on 4/1/25.
//

#ifdef LUISA_ENABLE_XIR

#include <luisa/runtime/rtx/ray.h>
#include <luisa/runtime/rtx/hit.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/dsl/rtx/ray_query.h>

#include "cuda_codegen_xir.h"

namespace luisa::compute::cuda {

CUDACodegenXIR::CUDACodegenXIR(StringScratch &scratch, bool allow_indirect) noexcept
    : _scratch{scratch}, _allow_indirect_dispatch{allow_indirect},
      _ray_type{Type::of<Ray>()},
      _triangle_hit_type{Type::of<TriangleHit>()},
      _procedural_hit_type{Type::of<ProceduralHit>()},
      _committed_hit_type{Type::of<CommittedHit>()},
      _ray_query_all_type{Type::of<RayQueryAll>()},
      _ray_query_any_type{Type::of<RayQueryAny>()},
      _indirect_buffer_type{Type::of<IndirectDispatchBuffer>()},
      _motion_srt_type{Type::of<MotionInstanceTransformSRT>()} {}

CUDACodegenXIR::~CUDACodegenXIR() noexcept = default;

void CUDACodegenXIR::emit(const xir::Module *module,
                          luisa::string_view device_lib,
                          luisa::string_view native_include) noexcept {
}

}// namespace luisa::compute::cuda

#endif