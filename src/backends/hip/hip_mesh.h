//
// Created by mike on 1/30/26.
//

#pragma once

#include <hip/hip_runtime.h>
#include <hiprt/hiprt.h>

#include <luisa/core/spin_mutex.h>
#include <luisa/runtime/rtx/mesh.h>

namespace luisa::compute::hip {

class HIPCommandEncoder;

class HIPMesh {

private:
    AccelOption _option;
    hiprtContext _hiprt_ctx{nullptr};
    hiprtGeometry _geometry{nullptr};
    hipDeviceptr_t _vertex_buffer{};
    size_t _vertex_buffer_size{};
    size_t _vertex_stride{};
    hipDeviceptr_t _triangle_buffer{};
    size_t _triangle_buffer_size{};
    mutable spin_mutex _mutex;

public:
    explicit HIPMesh(hiprtContext ctx, const AccelOption &option) noexcept;
    ~HIPMesh() noexcept;
    void build(HIPCommandEncoder &encoder, MeshBuildCommand *command) noexcept;
    [[nodiscard]] auto handle() const noexcept {
        std::scoped_lock lock{_mutex};
        return _geometry;
    }
    [[nodiscard]] auto option() const noexcept { return _option; }
};

}// namespace luisa::compute::hip
