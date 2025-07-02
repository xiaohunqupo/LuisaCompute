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
    size_t accelIndex;
    size_t meshIndex;
    static MeshHandle *AllocateHandle();
    static void DestroyHandle(MeshHandle *handle);
};
class Blas : public Resource {
    friend class Tlas;
private:
luisa::spin_mutex handleMtx;
    VkAccelerationStructureKHR _accel{nullptr};
    vstd::unique_ptr<DefaultBuffer> _accel_buffer;
    VkAccelerationStructureBuildGeometryInfoKHR *acceleration_build_geometry_info;
    AccelOption option;
    DefaultBuffer const *scratch_buffer{nullptr};
    uint64_t scratch_buffer_offset{0};
    vstd::fixed_vector<MeshHandle *, 2> handles;

    MeshHandle *add_accel_ref(Tlas *accel, uint index);
    void remove_accel_ref(MeshHandle *handle);
    void sync_tlas();
public:
    [[nodiscard]] auto &accel() const { return _accel; }
    Blas(Device *device, AccelOption const &option);
    void pre_build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    void build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    ~Blas();
    uint64_t get_accel_device_address() const;
};
}// namespace lc::vk