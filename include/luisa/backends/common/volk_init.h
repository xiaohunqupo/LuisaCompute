#pragma once

#include <volk.h>

#include <luisa/core/basic_types.h>
#include <luisa/core/logging.h>
#include <luisa/core/magic_enum.h>
#include <luisa/core/dynamic_module.h>

#ifndef LUISA_CHECK_VULKAN
#define LUISA_CHECK_VULKAN(x)                                            \
    do {                                                                 \
        auto ret = x;                                                    \
        if (ret != VK_SUCCESS) [[unlikely]] {                            \
            if (ret > 0 || ret == VK_ERROR_OUT_OF_DATE_KHR) [[likely]] { \
                LUISA_WARNING_WITH_LOCATION(                             \
                    "Vulkan call `" #x "` returned {} (code = {}).",     \
                    ::luisa::to_string(ret), luisa::to_underlying(ret)); \
            } else [[unlikely]] {                                        \
                LUISA_ERROR_WITH_LOCATION(                               \
                    "Vulkan call `" #x "` failed: {} (code = {}).",      \
                    ::luisa::to_string(ret), luisa::to_underlying(ret)); \
            }                                                            \
        }                                                                \
    } while (false)
#endif

namespace luisa::compute {

class VolkInitializer {

public:
    DynamicModule vk_module;
    void init(const luisa::filesystem::path &custom_path = {}, luisa::string_view lib_name = {}) {
        auto init_custom = [&] {
            auto ptr = vk_module.address("vkGetInstanceProcAddr");
            if (!ptr) [[unlikely]] {
                LUISA_ERROR("vkGetInstanceProcAddr symbol not found.");
            }
            volkInitializeCustom(reinterpret_cast<PFN_vkGetInstanceProcAddr>(ptr));
        };
        if (!custom_path.empty() && !lib_name.empty()) {
            vk_module = DynamicModule::load(custom_path, lib_name);
            if (!vk_module) [[unlikely]] {
                LUISA_ERROR("{} not found at {}.", dynamic_module_name(lib_name), luisa::to_string(custom_path));
            }
            init_custom();
        } else if (!custom_path.empty() && lib_name.empty()) {
#if defined(_WIN32)
            vk_module = DynamicModule::load(custom_path, "vulkan-1");
#elif defined(__APPLE__)
            do {
                vk_module = DynamicModule::load(custom_path, "libvulkan");
                if (vk_module) break;
                vk_module = DynamicModule::load(custom_path, "libvulkan.1");
                if (vk_module) break;
                vk_module = DynamicModule::load(custom_path, "libMoltenVK");
                if (vk_module) break;
                vk_module = DynamicModule::load(custom_path, "MoltenVK");
                if (vk_module) break;
                vk_module = DynamicModule::load(custom_path, "vulkan");
            } while (false);
#else
            vk_module = DynamicModule::load(custom_path, "libvulkan");
#endif
            if (!vk_module) [[unlikely]] {
                LUISA_ERROR("Vulkan lib not found at {}", luisa::to_string(custom_path));
            }
            init_custom();
        } else if (custom_path.empty() && !lib_name.empty()) {
            vk_module = DynamicModule::load(lib_name);
            if (!vk_module) [[unlikely]] {
                LUISA_ERROR("{} not found", dynamic_module_name(lib_name));
            }
            init_custom();
        } else {
            auto current_path = current_executable_path();
#if defined(_WIN32)
            vk_module = DynamicModule::load(current_path, "vulkan-1");
#elif defined(__APPLE__)
            do {
                vk_module = DynamicModule::load(current_path, "libvulkan");
                if (vk_module) break;
                vk_module = DynamicModule::load(current_path, "libvulkan.1");
                if (vk_module) break;
                vk_module = DynamicModule::load(current_path, "libMoltenVK");
                if (vk_module) break;
                vk_module = DynamicModule::load(current_path, "MoltenVK");
                if (vk_module) break;
                vk_module = DynamicModule::load(current_path, "vulkan");
            } while (false);
#else
            vk_module = DynamicModule::load(current_path, "libvulkan");
#endif
            if (vk_module) [[unlikely]] {
                init_custom();
            } else {
                LUISA_CHECK_VULKAN(volkInitialize());
            }
        }
    }
};

}// namespace luisa::compute