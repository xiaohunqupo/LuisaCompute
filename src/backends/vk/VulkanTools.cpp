/*
* Assorted commonly used Vulkan helper functions
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "VulkanTools.h"
#include "log.h"
#if !(defined(VK_USE_PLATFORM_IOS_MVK) || defined(VK_USE_PLATFORM_MACOS_MVK))
// iOS & macOS: VulkanExampleBase::getAssetPath() implemented externally to allow access to Objective-C components
const luisa::string get_asset_path()
{
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
	return "";
#elif defined(VK_EXAMPLE_DATA_DIR)
	return VK_EXAMPLE_DATA_DIR;
#else
	return "./../data/";
#endif
}
#endif

namespace vks
{
	namespace tools
	{
		bool error_mode_silent = false;

		luisa::string error_string(VkResult error_code)
		{
			switch (error_code)
			{
#define STR(r) case VK_ ##r: return #r
				STR(NOT_READY);
				STR(TIMEOUT);
				STR(EVENT_SET);
				STR(EVENT_RESET);
				STR(INCOMPLETE);
				STR(ERROR_OUT_OF_HOST_MEMORY);
				STR(ERROR_OUT_OF_DEVICE_MEMORY);
				STR(ERROR_INITIALIZATION_FAILED);
				STR(ERROR_DEVICE_LOST);
				STR(ERROR_MEMORY_MAP_FAILED);
				STR(ERROR_LAYER_NOT_PRESENT);
				STR(ERROR_EXTENSION_NOT_PRESENT);
				STR(ERROR_FEATURE_NOT_PRESENT);
				STR(ERROR_INCOMPATIBLE_DRIVER);
				STR(ERROR_TOO_MANY_OBJECTS);
				STR(ERROR_FORMAT_NOT_SUPPORTED);
				STR(ERROR_SURFACE_LOST_KHR);
				STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
				STR(SUBOPTIMAL_KHR);
				STR(ERROR_OUT_OF_DATE_KHR);
				STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
				STR(ERROR_VALIDATION_FAILED_EXT);
				STR(ERROR_INVALID_SHADER_NV);
#undef STR
			default:
				return "UNKNOWN_ERROR";
			}
		}

		luisa::string physical_device_type_string(VkPhysicalDeviceType type)
		{
			switch (type)
			{
#define STR(r) case VK_PHYSICAL_DEVICE_TYPE_ ##r: return #r
				STR(OTHER);
				STR(INTEGRATED_GPU);
				STR(DISCRETE_GPU);
				STR(VIRTUAL_GPU);
				STR(CPU);
#undef STR
			default: return "UNKNOWN_DEVICE_TYPE";
			}
		}

		VkBool32 get_supported_depth_format(VkPhysicalDevice physical_device, VkFormat *depth_format)
		{
			// Since all depth formats may be optional, we need to find a suitable depth format to use
			// Start with the highest precision packed format
			std::vector<VkFormat> depth_formats = {
				VK_FORMAT_D32_SFLOAT_S8_UINT,
				VK_FORMAT_D32_SFLOAT,
				VK_FORMAT_D24_UNORM_S8_UINT,
				VK_FORMAT_D16_UNORM_S8_UINT,
				VK_FORMAT_D16_UNORM
			};

			for (auto& format : depth_formats)
			{
				VkFormatProperties format_props;
				vkGetPhysicalDeviceFormatProperties(physical_device, format, &format_props);
				// Format must support depth stencil attachment for optimal tiling
				if (format_props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
				{
					*depth_format = format;
					return true;
				}
			}

			return false;
		}

		VkBool32 format_has_stencil(VkFormat format)
		{
			std::vector<VkFormat> stencil_formats = {
				VK_FORMAT_S8_UINT,
				VK_FORMAT_D16_UNORM_S8_UINT,
				VK_FORMAT_D24_UNORM_S8_UINT,
				VK_FORMAT_D32_SFLOAT_S8_UINT,
			};
			return std::find(stencil_formats.begin(), stencil_formats.end(), format) != std::end(stencil_formats);
		}

		// Returns if a given format support LINEAR filtering
		VkBool32 format_is_filterable(VkPhysicalDevice physical_device, VkFormat format, VkImageTiling tiling)
		{
			VkFormatProperties format_props;
			vkGetPhysicalDeviceFormatProperties(physical_device, format, &format_props);

			if (tiling == VK_IMAGE_TILING_OPTIMAL)
				return format_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

			if (tiling == VK_IMAGE_TILING_LINEAR)
				return format_props.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

			return false;
		}

		// Create an image memory barrier for changing the layout of
		// an image and put it into an active command buffer
		// See chapter 11.4 "Image Layout" for details

		void set_image_layout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageLayout old_image_layout,
			VkImageLayout new_image_layout,
			VkImageSubresourceRange subresource_range,
			VkPipelineStageFlags src_stage_mask,
			VkPipelineStageFlags dst_stage_mask)
		{
			// Create an image barrier object
			VkImageMemoryBarrier image_memory_barrier = vks::initializers::imageMemoryBarrier();
			image_memory_barrier.oldLayout = old_image_layout;
			image_memory_barrier.newLayout = new_image_layout;
			image_memory_barrier.image = image;
			image_memory_barrier.subresourceRange = subresource_range;

			// Source layouts (old)
			// Source access mask controls actions that have to be finished on the old layout
			// before it will be transitioned to the new layout
			switch (old_image_layout)
			{
			case VK_IMAGE_LAYOUT_UNDEFINED:
				// Image layout is undefined (or does not matter)
				// Only valid as initial layout
				// No flags required, listed only for completeness
				image_memory_barrier.srcAccessMask = 0;
				break;

			case VK_IMAGE_LAYOUT_PREINITIALIZED:
				// Image is preinitialized
				// Only valid as initial layout for linear images, preserves memory contents
				// Make sure host writes have been finished
				image_memory_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
				// Image is a color attachment
				// Make sure any writes to the color buffer have been finished
				image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
				// Image is a depth/stencil attachment
				// Make sure any writes to the depth/stencil buffer have been finished
				image_memory_barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
				// Image is a transfer source
				// Make sure any reads from the image have been finished
				image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				break;

			case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
				// Image is a transfer destination
				// Make sure any writes to the image have been finished
				image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
				// Image is read by a shader
				// Make sure any shader reads from the image have been finished
				image_memory_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
				break;
			default:
				// Other source layouts aren't handled (yet)
				break;
			}

			// Target layouts (new)
			// Destination access mask controls the dependency for the new image layout
			switch (new_image_layout)
			{
			case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
				// Image will be used as a transfer destination
				// Make sure any writes to the image have been finished
				image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
				// Image will be used as a transfer source
				// Make sure any reads from the image have been finished
				image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				break;

			case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
				// Image will be used as a color attachment
				// Make sure any writes to the color buffer have been finished
				image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
				// Image layout will be used as a depth/stencil attachment
				// Make sure any writes to depth/stencil buffer have been finished
				image_memory_barrier.dstAccessMask = image_memory_barrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
				break;

			case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
				// Image will be read in a shader (sampler, input attachment)
				// Make sure any writes to the image have been finished
				if (image_memory_barrier.srcAccessMask == 0)
				{
					image_memory_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
				}
				image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
				break;
			default:
				// Other source layouts aren't handled (yet)
				break;
			}

			// Put barrier inside setup command buffer
			vkCmdPipelineBarrier(
				cmdbuffer,
				src_stage_mask,
				dst_stage_mask,
				0,
				0, nullptr,
				0, nullptr,
				1, &image_memory_barrier);
		}

		// Fixed sub resource on first mip level and layer
		void set_image_layout(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkImageAspectFlags aspect_mask,
			VkImageLayout old_image_layout,
			VkImageLayout new_image_layout,
			VkPipelineStageFlags src_stage_mask,
			VkPipelineStageFlags dst_stage_mask)
		{
			VkImageSubresourceRange subresource_range = {};
			subresource_range.aspectMask = aspect_mask;
			subresource_range.baseMipLevel = 0;
			subresource_range.levelCount = 1;
			subresource_range.layerCount = 1;
			set_image_layout(cmdbuffer, image, old_image_layout, new_image_layout, subresource_range, src_stage_mask, dst_stage_mask);
		}

		void insert_image_memory_barrier(
			VkCommandBuffer cmdbuffer,
			VkImage image,
			VkAccessFlags src_access_mask,
			VkAccessFlags dst_access_mask,
			VkImageLayout old_image_layout,
			VkImageLayout new_image_layout,
			VkPipelineStageFlags src_stage_mask,
			VkPipelineStageFlags dst_stage_mask,
			VkImageSubresourceRange subresource_range)
		{
			VkImageMemoryBarrier image_memory_barrier = vks::initializers::imageMemoryBarrier();
			image_memory_barrier.srcAccessMask = src_access_mask;
			image_memory_barrier.dstAccessMask = dst_access_mask;
			image_memory_barrier.oldLayout = old_image_layout;
			image_memory_barrier.newLayout = new_image_layout;
			image_memory_barrier.image = image;
			image_memory_barrier.subresourceRange = subresource_range;

			vkCmdPipelineBarrier(
				cmdbuffer,
				src_stage_mask,
				dst_stage_mask,
				0,
				0, nullptr,
				0, nullptr,
				1, &image_memory_barrier);
		}

#if defined(__ANDROID__)
		// Android shaders are stored as assets in the apk
		// So they need to be loaded via the asset manager
		VkShaderModule load_shader(AAssetManager* asset_manager, const char *file_name, VkDevice device)
		{
			// Load shader from compressed asset
			AAsset* asset = AAssetManager_open(asset_manager, file_name, AASSET_MODE_STREAMING);
			assert(asset);
			size_t size = AAsset_getLength(asset);
			assert(size > 0);

			char *shader_code = new char[size];
			AAsset_read(asset, shader_code, size);
			AAsset_close(asset);

			VkShaderModule shader_module;
			VkShaderModuleCreateInfo module_create_info;
			module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			module_create_info.pNext = NULL;
			module_create_info.codeSize = size;
			module_create_info.pCode = (uint32_t*)shader_code;
			module_create_info.flags = 0;

			VK_CHECK_RESULT(vkCreateShaderModule(device, &module_create_info, NULL, &shader_module));

			delete[] shader_code;

			return shader_module;
		}
#else
		VkShaderModule load_shader(const char *file_name, VkDevice device)
		{
			std::ifstream is(file_name, std::ios::binary | std::ios::in | std::ios::ate);

			if (is.is_open())
			{
				size_t size = is.tellg();
				is.seekg(0, std::ios::beg);
				char* shader_code = new char[size];
				is.read(shader_code, size);
				is.close();

				assert(size > 0);

				VkShaderModule shader_module;
				VkShaderModuleCreateInfo module_create_info{};
				module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				module_create_info.codeSize = size;
				module_create_info.pCode = (uint32_t*)shader_code;

				VK_CHECK_RESULT(vkCreateShaderModule(device, &module_create_info, NULL, &shader_module));

				delete[] shader_code;

				return shader_module;
			}
			else
			{
				std::cerr << "Error: Could not open shader file \"" << file_name << "\"" << "\n";
				return VK_NULL_HANDLE;
			}
		}
#endif

		bool file_exists(const luisa::string &filename)
		{
			std::ifstream f(filename.c_str());
			return !f.fail();
		}

		uint32_t aligned_size(uint32_t value, uint32_t alignment)
        {
	        return (value + alignment - 1) & ~(alignment - 1);
        }

	}
}

