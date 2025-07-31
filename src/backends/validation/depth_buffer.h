#pragma once
#include "rw_resource.h"
namespace lc::validation {
class DepthBuffer : public RWResource {
public:
    DepthBuffer(uint64_t handle) : RWResource(handle, Tag::DEPTH_BUFFER, true) {}
    static constexpr luisa::string_view validation_res_name{"DepthBuffer"};
};
}// namespace lc::validation
