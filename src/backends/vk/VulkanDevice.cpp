/*
* Vulkan device class
*
* Encapsulates a physical Vulkan device and its logical representation
*
* Copyright (C) by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
// SRS - Enable beta extensions and make VK_KHR_portability_subset visible
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include "VulkanDevice.h"
#include <luisa/core/logging.h>
#include "log.h"
#include "device.h"
#include <luisa/core/dynamic_module.h>
#include <luisa/backends/common/volk_init.h>
namespace vks {
/**
	* Default constructor
	*
	* @param physical_device Physical device that is to be used
	*/
static luisa::spin_mutex g_volk_mtx;
static int32 g_volk_ref_count = 0;
static luisa::compute::VolkInitializer volk_initer;

void VulkanDevice::init_volk(luisa::filesystem::path const &custom_path, luisa::string_view lib_name) {
    std::lock_guard lck(g_volk_mtx);
    if (!volk_initer.vk_module) {
        volk_initer.init(custom_path, lib_name);
    }
}
VulkanDevice::VulkanDevice(VkPhysicalDevice physical_device) {
    {
        std::lock_guard lck(g_volk_mtx);
        ++g_volk_ref_count;
    }
    assert(physical_device);
    this->physical_device = physical_device;

    // Store Properties features, limits and properties of the physical device for later use
    // Device properties also contain limits and sparse properties
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    // Features should be checked by the examples before using them
    features_12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &features_12;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    // Memory properties are used regularly for creating all kinds of buffers
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    // Queue family properties, used for setting up requested queues upon device creation
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, nullptr);
    assert(queueFamilyCount > 0);
    queue_family_properties.resize(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queueFamilyCount, queue_family_properties.data());

    // Get list of supported extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extCount, nullptr);
    if (extCount > 0) {
        vstd::vector<VkExtensionProperties> extensions(extCount);
        if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extCount, &extensions.front()) == VK_SUCCESS) {
            for (auto ext : extensions) {
                supported_extensions.emplace(ext.extensionName);
            }
        }
    }
}

/** 
	* Default destructor
	*
	* @note Frees the logical device
	*/
VulkanDevice::~VulkanDevice() {
    if (logical_device) {
        vkDestroyDevice(logical_device, lc::vk::Device::alloc_callbacks());
    }
    {
        std::lock_guard lck(g_volk_mtx);
        if (--g_volk_ref_count == 0) {
            force_free_volk();
        }
    }
}
void VulkanDevice::force_free_volk() {
    volk_initer.vk_module.reset();
}

/**
	* Get the index of a memory type that has all the requested property bits set
	*
	* @param typeBits Bit mask with bits set for each memory type supported by the resource to request for (from VkMemoryRequirements)
	* @param properties Bit mask of properties for the memory type to request
	* @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
	* 
	* @return Index of the requested memory type
	*
	* @throw Throws an exception if memTypeFound is null and no memory type could be found that supports the requested properties
	*/
uint32_t VulkanDevice::get_memory_type(uint32_t type_bits, VkMemoryPropertyFlags properties, VkBool32 *mem_type_found) const {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((type_bits & 1) == 1) {
            if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                if (mem_type_found) {
                    *mem_type_found = true;
                }
                return i;
            }
        }
        type_bits >>= 1;
    }

    if (mem_type_found) {
        *mem_type_found = false;
        return 0;
    } else {
        LUISA_ERROR("Could not find a matching memory type");
    }
}

/**
	* Get the index of a queue family that supports the requested queue flags
	* SRS - support VkQueueFlags parameter for requesting multiple flags vs. VkQueueFlagBits for a single flag only
	*
	* @param queueFlags Queue flags to find a queue family index for
	*
	* @return Index of the queue family index that matches the flags
	*
	* @throw Throws an exception if no queue family index could be found that supports the requested flags
	*/
uint32_t VulkanDevice::get_queue_family_index(VkQueueFlags queueFlags) const {
    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if ((queueFlags & VK_QUEUE_COMPUTE_BIT) == queueFlags) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++) {
            if ((queue_family_properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)) {
                return i;
            }
        }
    }

    // Dedicated queue for transfer
    // Try to find a queue family index that supports transfer but not graphics and compute
    if ((queueFlags & VK_QUEUE_TRANSFER_BIT) == queueFlags) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++) {
            if ((queue_family_properties[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && ((queue_family_properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((queue_family_properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)) {
                return i;
            }
        }
    }

    // For other queue types or if no separate compute queue is present, return the first one to support the requested flags
    for (uint32_t i = 0; i < static_cast<uint32_t>(queue_family_properties.size()); i++) {
        if ((queue_family_properties[i].queueFlags & queueFlags) == queueFlags) {
            return i;
        }
    }

    LUISA_ERROR("Could not find a matching queue family index");
}

/**
	* Create the logical device based on the assigned physical device, also gets default queue family indices
	*
	* @param enabled_features Can be used to enable certain features upon device creation
	* @param pNextChain Optional chain of pointer to extension structures
	* @param useSwapChain Set to false for headless rendering to omit the swapchain device extensions
	* @param requestedQueueTypes Bit flags specifying the queue types to be requested from the device  
	*
	* @return VkResult of the device creation call
	*/
VkResult VulkanDevice::create_logical_device(VkPhysicalDeviceFeatures &enabled_features, vstd::span<const vstd::string> enabled_extensions, void *p_next_chain, bool use_swap_chain, VkQueueFlags requested_queue_types) {
    // Desired queues need to be requested upon logical device creation
    // Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
    // requests different queue types

    vstd::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};

    // Get queue family indices for the requested queue family types
    // Note that the indices may overlap depending on the implementation

    const float defaultQueuePriority(1.0f);
    const float computedefaultQueuePriority(0.5f);
    const float copydefaultQueuePriority(0.0f);

    // Graphics queue
    if (requested_queue_types & VK_QUEUE_GRAPHICS_BIT) {
        queue_family_indices.graphics = get_queue_family_index(VK_QUEUE_GRAPHICS_BIT);
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queue_family_indices.graphics;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &defaultQueuePriority;
        queueCreateInfos.push_back(queueInfo);
    } else {
        queue_family_indices.graphics = 0;
    }

    // Dedicated compute queue
    if (requested_queue_types & VK_QUEUE_COMPUTE_BIT) {
        queue_family_indices.compute = get_queue_family_index(VK_QUEUE_COMPUTE_BIT);
        if (queue_family_indices.compute != queue_family_indices.graphics) {
            // If compute family index differs, we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = queue_family_indices.compute;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &computedefaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    } else {
        // Else we use the same queue
        queue_family_indices.compute = queue_family_indices.graphics;
    }

    // Dedicated transfer queue
    if (requested_queue_types & VK_QUEUE_TRANSFER_BIT) {
        queue_family_indices.transfer = get_queue_family_index(VK_QUEUE_TRANSFER_BIT);
        if ((queue_family_indices.transfer != queue_family_indices.graphics) && (queue_family_indices.transfer != queue_family_indices.compute)) {
            // If transfer family index differs, we need an additional queue create info for the transfer queue
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = queue_family_indices.transfer;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &copydefaultQueuePriority;
            queueCreateInfos.push_back(queueInfo);
        }
    } else {
        // Else we use the same queue
        queue_family_indices.transfer = queue_family_indices.graphics;
    }

    // Create the logical device representation
    vstd::vector<const char *> deviceExtensions;
    vstd::push_back_func(
        deviceExtensions,
        enabled_extensions.size(),
        [&](size_t i) {
            return enabled_extensions[i].c_str();
        });

    if (use_swap_chain) {
        // If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures = &enabled_features;

    // If a pNext(Chain) has been passed, we need to add it to the device creation info
    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
    if (p_next_chain) {
        physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        physicalDeviceFeatures2.features = enabled_features;
        physicalDeviceFeatures2.pNext = p_next_chain;
        deviceCreateInfo.pEnabledFeatures = nullptr;
        deviceCreateInfo.pNext = &physicalDeviceFeatures2;
    }

    // Enable the debug marker extension if it is present (likely meaning a debugging tool is present)
    if (extension_supported(VK_EXT_DEBUG_MARKER_EXTENSION_NAME)) {
        deviceExtensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
        enable_debug_markers = true;
    }

#if (defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK)) && defined(VK_KHR_portability_subset)
    // SRS - When running on iOS/macOS with MoltenVK and VK_KHR_portability_subset is defined and supported by the device, enable the extension
    if (extension_supported(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif

    if (deviceExtensions.size() > 0) {
        for (const char *enabledExtension : deviceExtensions) {
            if (!extension_supported(enabledExtension)) {
                LUISA_ERROR("Enabled device extension \"{}\" is not present at device level", enabledExtension);
            }
        }

        deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    }

    this->enabled_features = enabled_features;
    if (!logical_device) {
        VkResult result = vkCreateDevice(physical_device, &deviceCreateInfo, lc::vk::Device::alloc_callbacks(), &logical_device);
        if (result != VK_SUCCESS) {
            return result;
        }

        // Create a default command pool for graphics command buffers

        return result;
    }
    return VK_SUCCESS;
}

bool VulkanDevice::extension_supported(vstd::string_view extension) {
    return supported_extensions.find(extension) != supported_extensions.end();
}

VkFormat VulkanDevice::get_supported_depth_format(bool check_sampling_support) {
    // All depth formats may be optional, so we need to find a suitable depth format to use
    vstd::vector<VkFormat> depthFormats = {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D16_UNORM};
    for (auto &format : depthFormats) {
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physical_device, format, &formatProperties);
        // Format must support depth stencil attachment for optimal tiling
        if (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            if (check_sampling_support) {
                if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
                    continue;
                }
            }
            return format;
        }
    }
    LUISA_ERROR("Could not find a matching depth format");
}

};// namespace vks
