#include "fallback_prim.h"

namespace luisa::compute::fallback {

FallbackPrim::FallbackPrim(RTCDevice device, RTCGeometryType geom_type, const AccelOption &option) noexcept
    : _handle{rtcNewScene(device)}, _geometry{rtcNewGeometry(device, geom_type)}, _geom_type{geom_type} {
    luisa_fallback_accel_set_flags(_handle, option);
    rtcSetGeometryMask(_geometry, ~0u);
    rtcAttachGeometry(_handle, _geometry);
    rtcReleaseGeometry(_geometry);// already moved into the scene
}

FallbackPrim::~FallbackPrim() noexcept {
    rtcReleaseScene(_handle);
}

}// namespace luisa::compute::fallback
