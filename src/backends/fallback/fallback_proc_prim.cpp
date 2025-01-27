#include <luisa/core/logging.h>

#include "fallback_device_api.h"
#include "fallback_buffer.h"
#include "fallback_proc_prim.h"

namespace luisa::compute::fallback {

FallbackProceduralPrim::FallbackProceduralPrim(RTCDevice device, const AccelOption &option) noexcept
    : FallbackPrim{device, RTC_GEOMETRY_TYPE_USER, option} {}

void FallbackProceduralPrim::build(luisa::unique_ptr<ProceduralPrimitiveBuildCommand> cmd) noexcept {
    auto aabb_buffer = reinterpret_cast<FallbackBuffer *>(cmd->aabb_buffer())->data();
    LUISA_DEBUG_ASSERT(cmd->aabb_buffer_size() % sizeof(AABB) == 0u, "Invalid AABB buffer size.");
    auto aabb_count = cmd->aabb_buffer_size() / sizeof(AABB);
    rtcSetGeometryUserPrimitiveCount(geometry(), aabb_count);
    rtcSetGeometryUserData(geometry(), aabb_buffer + cmd->aabb_buffer_offset());
    rtcSetGeometryBoundsFunction(
        geometry(), [](const RTCBoundsFunctionArguments *args) noexcept {
            auto aabb_buffer = static_cast<const AABB *>(args->geometryUserPtr);
            auto aabb = aabb_buffer[args->primID];
            *args->bounds_o = {
                .lower_x = aabb.packed_min[0],
                .lower_y = aabb.packed_min[1],
                .lower_z = aabb.packed_min[2],
                .align0 = 0.f,
                .upper_x = aabb.packed_max[0],
                .upper_y = aabb.packed_max[1],
                .upper_z = aabb.packed_max[2],
                .align1 = 0.f,
            };
            // TODO: support motion
        },
        nullptr);
    rtcSetGeometryIntersectFunction(geometry(), reinterpret_cast<RTCIntersectFunctionN>(api::luisa_fallback_ray_query_procedural_intersect_function));
    rtcSetGeometryOccludedFunction(geometry(), reinterpret_cast<RTCOccludedFunctionN>(api::luisa_fallback_ray_query_procedural_occluded_function));
    rtcCommitGeometry(geometry());
    rtcCommitScene(handle());
}

}// namespace luisa::compute::fallback
