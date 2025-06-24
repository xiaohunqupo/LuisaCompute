#include "vk_func_table.h"
#include "device.h"
namespace lc::vk {
#define SET_VK_FUNC(func) func = (PFN_##func)vkGetInstanceProcAddr(device->instance(), #func)

void VkFuncTable::init(Device *device) {
    SET_VK_FUNC(vkGetAccelerationStructureBuildSizesKHR);
    SET_VK_FUNC(vkCreateAccelerationStructureKHR);
    SET_VK_FUNC(vkCmdBuildAccelerationStructuresKHR);
    SET_VK_FUNC(vkDestroyAccelerationStructureKHR);
    SET_VK_FUNC(vkGetAccelerationStructureDeviceAddressKHR);
}
VkFuncTable::VkFuncTable() {}
VkFuncTable::~VkFuncTable() {}
#undef SET_VK_FUNC
}// namespace lc::vk