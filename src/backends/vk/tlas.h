#pragma once
#include <vulkan/vulkan_core.h>
#include "resource.h"
#include "default_buffer.h"
#include <luisa/runtime/rtx/accel.h>
namespace lc::vk {
class CommandBuffer;
using namespace luisa;
using namespace luisa::compute;
class MeshHandle;
class Blas;
class Tlas : public Resource {
    friend class Blas;
private:
    VkAccelerationStructureKHR _accel{nullptr};
    vstd::unique_ptr<DefaultBuffer> _accel_buffer;
    vstd::unique_ptr<DefaultBuffer> _instance_buffer;
    VkAccelerationStructureBuildGeometryInfoKHR *acceleration_build_geometry_info;
    AccelOption option;
    DefaultBuffer const *scratch_buffer{nullptr};
    vstd::unordered_map<uint64, MeshHandle *> set_map;
    uint64_t scratch_buffer_offset{0};
    uint _last_instance_count = 0;
    struct Instance {
        MeshHandle *handle = nullptr;
    };
    bool require_rebuild = true;
    vstd::vector<Instance> allInstance;
    void resize_instance(size_t size);
    void update_mesh(MeshHandle *handle);
    void set_mesh(Blas *mesh, uint64 index);

public:
    Tlas(Device *device, AccelOption const &option);
    void pre_build(
        CommandBuffer &cmdbuffer,
        uint instance_count,
        luisa::vector<VkWriteDescriptorSet> &write_desc_sets,
        luisa::vector<uint4> &cache,
        luisa::span<AccelBuildCommand::Modification const> modifications,
        AccelBuildRequest request);
    void build(
        CommandBuffer &cmdbuffer,
        uint instance_count);
    ~Tlas();
    [[nodiscard]] auto &accel() const { return _accel; }
    [[nodiscard]] auto instance_buffer() const { return _instance_buffer.get(); }
    [[nodiscard]] auto accel_buffer() const { return _accel_buffer.get(); }
};
}// namespace lc::vk