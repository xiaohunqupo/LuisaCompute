#pragma once

#include "fallback_embree.h"

namespace luisa::compute::fallback {

class FallbackPrim {

private:
    RTCScene _handle;
    RTCGeometry _geometry;
    RTCGeometryType _geom_type;

public:
    FallbackPrim(RTCDevice device, RTCGeometryType geom_type, const AccelOption &option) noexcept;
    virtual ~FallbackPrim() noexcept;
    [[nodiscard]] auto type() const noexcept { return _geom_type; }
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto geometry() const noexcept { return _geometry; }
};

}// namespace luisa::compute::fallback
