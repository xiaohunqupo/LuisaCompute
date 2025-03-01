#include <luisa/core/logging.h>

#include "fallback_buffer.h"
#include "fallback_curve.h"

namespace luisa::compute::fallback {

FallbackCurve::FallbackCurve(RTCDevice device, const AccelOption &option) noexcept
    : FallbackPrim{device, option} {}

void FallbackCurve::build(luisa::unique_ptr<CurveBuildCommand> cmd) noexcept {
    // create the embree geometry if not created
    if (std::scoped_lock lock{_geom_mutex}; !_is_geometry_created()) {
        _create_geometry([basis = cmd->basis()] {
            switch (basis) {
                case CurveBasis::PIECEWISE_LINEAR: return RTC_GEOMETRY_TYPE_ROUND_LINEAR_CURVE;
                case CurveBasis::CUBIC_BSPLINE: return RTC_GEOMETRY_TYPE_ROUND_BSPLINE_CURVE;
                case CurveBasis::CATMULL_ROM: return RTC_GEOMETRY_TYPE_ROUND_CATMULL_ROM_CURVE;
                case CurveBasis::BEZIER: return RTC_GEOMETRY_TYPE_ROUND_BEZIER_CURVE;
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid curve basis 0x{:x}.", luisa::to_underlying(basis));
        }());
    }
    // build the geometry
    auto cp_buffer = reinterpret_cast<FallbackBuffer *>(cmd->cp_buffer())->data();
    auto seg_buffer = reinterpret_cast<FallbackBuffer *>(cmd->seg_buffer())->data();
    auto cp_buffer_offset = cmd->cp_buffer_offset();
    auto seg_buffer_offset = cmd->seg_buffer_offset();
    rtcSetSharedGeometryBuffer(geometry(), RTC_BUFFER_TYPE_VERTEX, 0u, RTC_FORMAT_FLOAT4,
                               cp_buffer, cp_buffer_offset, cmd->cp_stride(), cmd->cp_count());
    rtcSetSharedGeometryBuffer(geometry(), RTC_BUFFER_TYPE_INDEX, 0u, RTC_FORMAT_UINT,
                               seg_buffer, seg_buffer_offset, sizeof(uint), cmd->seg_count());
    rtcCommitGeometry(geometry());
    rtcCommitScene(handle());
}

}// namespace luisa::compute::fallback
