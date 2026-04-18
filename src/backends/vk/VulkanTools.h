/*
* Assorted Vulkan helper functions
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <volk.h>
#include "VulkanInitializers.hpp"
#include <luisa/core/stl/string.h>

#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#if defined(_WIN32)
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#elif defined(__ANDROID__)
#include "VulkanAndroid.h"
#include <android/asset_manager.h>
#endif

// Custom define for better code readability
#define VK_FLAGS_NONE 0
// Default fence timeout in nanoseconds
#define DEFAULT_FENCE_TIMEOUT 100000000000

// Macro to check and display Vulkan return results

const luisa::string get_asset_path();

namespace vks
{
	namespace tools
	{
		/** @brief Disable message boxes on fatal errors */
		extern bool error_mode_silent;

		/** @brief Returns an error code as a string */
		luisa::string error_string(VkResult error_code);

		/** @brief Returns the device type as a string */
		luisa::string physical_device_type_string(VkPhysicalDeviceType type);

		// Selected a suitable supported depth format starting with 32 bit down to 16 bit
		// Returns false if none of the depth formats in the list is supported by the device
		VkBool32 get_supported_depth_format(VkPhysicalDevice physical_device, VkFormat *depth_format);

		// Returns true if a given format supports LINEAR filtering
		VkBool32 format_is_filterable(VkPhysicalDevice physical_device, VkFormat format, VkImageTiling tiling);
		// Returns true if a given format has a stencil part
		VkBool32 format_has_stencil(VkFormat format);

		// Put an image memory barrier for setting an image layout on the sub resource into the given command buffer
		void set_image_layout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageLayout old_image_layout,
			VkImageLayout new_image_layout,
			VkImageSubresourceRange subresource_range,
			VkPipelineStageFlags src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dst_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
		// Uses a fixed sub resource layout with first mip level and layer
		void set_image_layout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageAspectFlags aspect_mask,
			VkImageLayout old_image_layout,
			VkImageLayout new_image_layout,
			VkPipelineStageFlags src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VkPipelineStageFlags dst_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

		/** @brief Insert an image memory barrier into the command buffer */
		void insert_image_memory_barrier(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkAccessFlags src_access_mask,
			VkAccessFlags dst_access_mask,
			VkImageLayout old_image_layout,
			VkImageLayout new_image_layout,
			VkPipelineStageFlags src_stage_mask,
			VkPipelineStageFlags dst_stage_mask,
			VkImageSubresourceRange subresource_range);

		// Load a SPIR-V shader (binary)
#if defined(__ANDROID__)
		VkShaderModule load_shader(AAssetManager* asset_manager, const char *file_name, VkDevice device);
#else
		VkShaderModule load_shader(const char *file_name, VkDevice device);
#endif

		/** @brief Checks if a file exists */
		bool file_exists(const luisa::string &filename);

		uint32_t aligned_size(uint32_t value, uint32_t alignment);
	}
}

