#pragma once

#include "fallback_embree.h"

namespace luisa::compute::fallback {

class FallbackPrim {

private:
    RTCScene _handle;
    RTCGeometry _geometry;
    AccelMotionOption _motion;

protected:
    [[nodiscard]] bool _is_geometry_created() const noexcept { return _geometry != nullptr; }
    void _create_geometry(RTCGeometryType geom_type) noexcept;

public:
    FallbackPrim(RTCDevice device, RTCGeometryType geom_type, const AccelOption &option) noexcept;
    // FIXME: workaround for curve that cannot decide geometry type on construction
    FallbackPrim(RTCDevice device, const AccelOption &option) noexcept;
    [[nodiscard]] virtual bool is_curve() const noexcept { return false; }
    virtual ~FallbackPrim() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto geometry() const noexcept { return _geometry; }
    [[nodiscard]] auto motion() const noexcept { return _motion; }
};

}// namespace luisa::compute::fallback
