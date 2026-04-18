#pragma once
#include <volk.h>
#include "resource.h"
#include "default_buffer.h"
#include <luisa/runtime/rtx/accel.h>
namespace lc::vk {
class CommandBuffer;
using namespace luisa;
using namespace luisa::compute;
class Tlas;
class Blas;
class MeshHandle {
    friend class Tlas;
public:
    Blas *mesh;
    Tlas *accel;
    size_t accel_index;
    size_t mesh_index;
    static MeshHandle *allocate_handle();
    static void destroy_handle(MeshHandle *handle);
};
class Blas : public Resource {
    friend class Tlas;
private:
    luisa::spin_mutex _handle_mtx;
    VkAccelerationStructureKHR _accel{nullptr};
    vstd::unique_ptr<DefaultBuffer> _accel_buffer;
    VkAccelerationStructureBuildGeometryInfoKHR *_acceleration_build_geometry_info;
    AccelOption _option;
    Buffer const *_scratch_buffer{nullptr};
    uint64_t _scratch_buffer_offset{0};
    vstd::fixed_vector<MeshHandle *, 2> _handles;

    MeshHandle *_add_accel_ref(Tlas *accel, uint index);
    void _remove_accel_ref(MeshHandle *handle);
    void _sync_tlas();
    void _pre_build(
        CommandBuffer &cmdbuffer,
        VkAccelerationStructureGeometryKHR* acceleration_structure_geometry,
        uint32_t primitive_count,
        AccelBuildRequest request
    );
public:
    [[nodiscard]] auto &accel() const { return _accel; }
    Blas(Device *device, AccelOption const &option);
    void pre_build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    void pre_build(
        CommandBuffer &cmdbuffer,
        ProceduralPrimitiveBuildCommand const *cmd);
    void build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    void build(
        CommandBuffer &cmdbuffer,
        ProceduralPrimitiveBuildCommand const *cmd);
    ~Blas();
    uint64_t get_accel_device_address() const;
};
}// namespace lc::vk