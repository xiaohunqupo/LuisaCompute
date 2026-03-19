//
// Created by mike on 1/30/26.
//

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"
#include "hip_stream.h"
#include "hip_device.h"
#include "hip_mesh.h"

namespace luisa::compute::hip {

HIPMesh::HIPMesh(hiprtContext ctx, const AccelOption &option) noexcept
    : _option{option}, _hiprt_ctx{ctx} {}

HIPMesh::~HIPMesh() noexcept {
    if (_geometry) {
        hiprtDestroyGeometry(_hiprt_ctx, _geometry);
    }
}

void HIPMesh::build(HIPCommandEncoder &encoder, MeshBuildCommand *command) noexcept {

    auto vertex_buffer = reinterpret_cast<const HIPBuffer *>(command->vertex_buffer());
    auto triangle_buffer = reinterpret_cast<const HIPBuffer *>(command->triangle_buffer());
    LUISA_ASSERT(command->vertex_buffer_offset() + command->vertex_buffer_size() <= vertex_buffer->size_bytes(),
                 "Vertex buffer offset + size exceeds buffer size {}.", vertex_buffer->size_bytes());
    LUISA_ASSERT(command->triangle_buffer_offset() + command->triangle_buffer_size() <= triangle_buffer->size_bytes(),
                 "Triangle buffer offset + size exceeds buffer size {}.", triangle_buffer->size_bytes());

    std::scoped_lock lock{_mutex};

    auto requires_build =
        _geometry == nullptr ||
        !_option.allow_update ||
        command->request() == AccelBuildRequest::FORCE_BUILD ||
        vertex_buffer->handle() + command->vertex_buffer_offset() != _vertex_buffer ||
        command->vertex_buffer_size() != _vertex_buffer_size ||
        command->vertex_stride() != _vertex_stride ||
        triangle_buffer->handle() + command->triangle_buffer_offset() != _triangle_buffer ||
        command->triangle_buffer_size() != _triangle_buffer_size;

    _vertex_buffer = vertex_buffer->handle() + command->vertex_buffer_offset();
    _vertex_buffer_size = command->vertex_buffer_size();
    _vertex_stride = command->vertex_stride();
    _triangle_buffer = triangle_buffer->handle() + command->triangle_buffer_offset();
    _triangle_buffer_size = command->triangle_buffer_size();

    auto vertex_count = static_cast<uint32_t>(_vertex_buffer_size / _vertex_stride);
    auto triangle_count = static_cast<uint32_t>(_triangle_buffer_size / sizeof(Triangle));

    hiprtTriangleMeshPrimitive mesh_prim{};
    mesh_prim.vertices = reinterpret_cast<hiprtDevicePtr>(_vertex_buffer);
    mesh_prim.vertexCount = vertex_count;
    mesh_prim.vertexStride = static_cast<uint32_t>(_vertex_stride);
    mesh_prim.triangleIndices = reinterpret_cast<hiprtDevicePtr>(_triangle_buffer);
    mesh_prim.triangleCount = triangle_count;
    mesh_prim.triangleStride = sizeof(Triangle);

    hiprtGeometryBuildInput build_input{};
    build_input.type = hiprtPrimitiveTypeTriangleMesh;
    build_input.primitive.triangleMesh = mesh_prim;

    hiprtBuildOptions build_options{};
    build_options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    auto hip_stream = encoder.stream()->handle();

    if (requires_build) {
        if (_geometry) {
            hiprtDestroyGeometry(_hiprt_ctx, _geometry);
            _geometry = nullptr;
        }
        LUISA_CHECK_HIPRT(hiprtCreateGeometry(_hiprt_ctx, build_input, build_options, _geometry));

        size_t temp_size = 0;
        LUISA_CHECK_HIPRT(hiprtGetGeometryBuildTemporaryBufferSize(_hiprt_ctx, build_input, build_options, temp_size));

        hipDeviceptr_t temp_buffer{};
        if (temp_size > 0) {
            LUISA_CHECK_HIP(hipMallocAsync(reinterpret_cast<void **>(&temp_buffer), temp_size, hip_stream));
        }

        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationBuild,
                                             build_input, build_options,
                                             temp_buffer, hip_stream, _geometry));

        if (temp_buffer) {
            LUISA_CHECK_HIP(hipFreeAsync(reinterpret_cast<void *>(temp_buffer), hip_stream));
        }
    } else {
        LUISA_CHECK_HIPRT(hiprtBuildGeometry(_hiprt_ctx, hiprtBuildOperationUpdate,
                                             build_input, build_options,
                                             nullptr, hip_stream, _geometry));
    }
}

}// namespace luisa::compute::hip
