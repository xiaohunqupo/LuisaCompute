#pragma once
#include <vulkan/vulkan_core.h>
namespace lc::vk {
class Device;
#define DEFINE_VK_FUNC(func) PFN_##func func
class VkFuncTable {
public:
    DEFINE_VK_FUNC(vkGetAccelerationStructureBuildSizesKHR);
    DEFINE_VK_FUNC(vkCreateAccelerationStructureKHR);
    DEFINE_VK_FUNC(vkCmdBuildAccelerationStructuresKHR);
    DEFINE_VK_FUNC(vkDestroyAccelerationStructureKHR);
    VkFuncTable();
    void init(Device *device);
    ~VkFuncTable();
};
}// namespace lc::vk
#undef DEFINE_VK_FUNC