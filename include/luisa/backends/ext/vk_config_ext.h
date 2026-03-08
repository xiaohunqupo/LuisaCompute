#pragma once
#include <vulkan/vulkan_core.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/string.h>
#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/backends/ext/vk_custom_cmd.h>
struct IDxcCompiler3;
struct IDxcLibrary;
struct IDxcUtils;
namespace luisa::compute {

class VulkanDeviceConfigExt : public DeviceConfigExt {
public:
    struct ExternalDevice {
        VkInstance instance{};
        VkPhysicalDevice physical_device{};
        VkDevice device{};
        VkQueue graphics_queue{};
        VkQueue compute_queue{};
        VkQueue copy_queue{};
    };
    struct VulkanLibPath {
        luisa::filesystem::path lib_path;
        luisa::string lib_name;
    };
    VulkanDeviceConfigExt() noexcept = default;
    ~VulkanDeviceConfigExt() noexcept = default;
    [[nodiscard]] virtual ExternalDevice create_external_device() noexcept {
        return {};
    }
    [[nodiscard]] virtual bool enable_bindless_feature() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_raytracing_feature() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_interop_feature() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_device_address_feature() const noexcept {
        return true;
    }
    [[nodiscard]] virtual bool enable_surface_feature() const noexcept {
        return true;
    }
    virtual VkCommandBuffer borrow_command_buffer(
        StreamTag stream_tag) noexcept { return nullptr; }
    virtual VulkanLibPath external_vulkan_lib_path() noexcept { return {}; }
    virtual bool execute_command_buffer(VkCommandBuffer cmd_buffer) noexcept { return false; }
    virtual bool signal_semaphore(VkQueue queue, VkSemaphore _semaphore, uint64_t index) noexcept { return false; }
    virtual bool wait_semaphore(VkQueue queue, VkSemaphore _semaphore, uint64_t index) noexcept { return false; }
    virtual bool sync_semaphore(VkSemaphore _semaphore, uint64_t index) noexcept { return false; }
    virtual bool load_dxc() const noexcept { return true; }
    virtual void init_volk(PFN_vkGetInstanceProcAddr handler) noexcept {}
    virtual luisa::vector<luisa::string> extra_instance_exts() noexcept { return {}; }
    virtual luisa::vector<luisa::string> extra_device_exts() noexcept { return {}; }
    virtual void get_defragment_function(luisa::move_only_function<void()> &&defragment_func) {}
    virtual void readback_vulkan_device(
        VkInstance instance,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkAllocationCallbacks *alloc_callback,
        VkPipelineCacheHeaderVersionOne const &pso_meta,
        VkQueue graphics_queue,
        VkQueue compute_queue,
        VkQueue copy_queue,
        uint32_t graphics_queue_family_index,
        uint32_t compute_queue_family_index,
        uint32_t copy_queue_family_index,
        IDxcCompiler3 *dxc_compiler,
        IDxcLibrary *dxc_library,
        IDxcUtils *dxc_utils) noexcept {}
    virtual luisa::span<VKCustomCmd::ResourceUsage const> before_states(uint64_t stream_handle) noexcept { return {}; }
    virtual luisa::span<VKCustomCmd::ResourceUsage const> after_states(uint64_t stream_handle) noexcept { return {}; }
    // return VkPhysicalDeviceXXXFeatures*
    virtual void *device_feature_settings() noexcept { return nullptr; }
};
}// namespace luisa::compute