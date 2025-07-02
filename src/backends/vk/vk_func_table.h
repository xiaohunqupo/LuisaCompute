#pragma once
#include <vulkan/vulkan.h>
namespace lc::vk {
class Device;
#define DEFINE_VK_FUNC(func) PFN_##func func
class VkFuncTable {
public:
    DEFINE_VK_FUNC(vkGetAccelerationStructureBuildSizesKHR);
    DEFINE_VK_FUNC(vkCreateAccelerationStructureKHR);
    DEFINE_VK_FUNC(vkCmdBuildAccelerationStructuresKHR);
    DEFINE_VK_FUNC(vkDestroyAccelerationStructureKHR);
    DEFINE_VK_FUNC(vkGetAccelerationStructureDeviceAddressKHR);

    DEFINE_VK_FUNC(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    DEFINE_VK_FUNC(vkGetPhysicalDeviceSurfaceFormatsKHR);
    DEFINE_VK_FUNC(vkGetPhysicalDeviceSurfacePresentModesKHR);
    DEFINE_VK_FUNC(vkCreateSwapchainKHR);
    DEFINE_VK_FUNC(vkGetSwapchainImagesKHR);
    DEFINE_VK_FUNC(vkDestroySwapchainKHR);
    DEFINE_VK_FUNC(vkDestroySurfaceKHR);

#if defined(LUISA_PLATFORM_WINDOWS)
    DEFINE_VK_FUNC(vkCreateWin32SurfaceKHR);
#elif defined(LUISA_PLATFORM_APPLE)
    DEFINE_VK_FUNC(vkCreateMacOSSurfaceMVK);
#else
#if LUISA_ENABLE_WAYLAND
#else
#endif
#endif
    VkFuncTable();
    void init(Device *device);
    ~VkFuncTable();
};
}// namespace lc::vk
#undef DEFINE_VK_FUNC