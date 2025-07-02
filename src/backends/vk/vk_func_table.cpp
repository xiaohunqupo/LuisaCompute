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

    SET_VK_FUNC(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    SET_VK_FUNC(vkGetPhysicalDeviceSurfaceFormatsKHR);
    SET_VK_FUNC(vkGetPhysicalDeviceSurfacePresentModesKHR);
    SET_VK_FUNC(vkCreateSwapchainKHR);
    SET_VK_FUNC(vkGetSwapchainImagesKHR);
    SET_VK_FUNC(vkDestroySwapchainKHR);
    SET_VK_FUNC(vkDestroySurfaceKHR);

#if defined(LUISA_PLATFORM_WINDOWS)
    SET_VK_FUNC(vkCreateWin32SurfaceKHR);
#elif defined(LUISA_PLATFORM_APPLE)
    SET_VK_FUNC(vkCreateMacOSSurfaceMVK);
#else
#if LUISA_ENABLE_WAYLAND
#else
#endif
#endif
}
VkFuncTable::VkFuncTable() {}
VkFuncTable::~VkFuncTable() {}
#undef SET_VK_FUNC
}// namespace lc::vk