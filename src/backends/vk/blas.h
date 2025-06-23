#pragma once
#include <vulkan/vulkan_core.h>
#include "resource.h"
#include "default_buffer.h"
#include <luisa/runtime/rtx/accel.h>
namespace lc::vk {
class CommandBuffer;
using namespace luisa;
using namespace luisa::compute;
class Blas : public Resource {
private:
    VkAccelerationStructureKHR _accel{nullptr};
    vstd::unique_ptr<DefaultBuffer> _accel_buffer;
    VkAccelerationStructureBuildGeometryInfoKHR *acceleration_build_geometry_info;
    AccelOption option;
    DefaultBuffer const *scratch_buffer{nullptr};
    uint64_t scratch_buffer_offset{0};

public:
    [[nodiscard]] auto accel() const { return _accel; }
    Blas(Device *device, AccelOption const &option);
    void pre_build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    void build(
        CommandBuffer &cmdbuffer,
        MeshBuildCommand const *cmd);
    ~Blas();
};
}// namespace lc::vk