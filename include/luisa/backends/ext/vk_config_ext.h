#pragma once
#include <vulkan/vulkan_core.h>
#include <luisa/runtime/rhi/device_interface.h>
struct IDxcCompiler3;
struct IDxcLibrary;
struct IDxcUtils;
namespace luisa::compute {

class VulkanDeviceConfigExt : public DeviceConfigExt {
public:
    VulkanDeviceConfigExt() = default;
    ~VulkanDeviceConfigExt() = default;
    [[nodiscard]] virtual bool enable_fallback() const {
        return false;
    }
    virtual VkCommandBuffer borrow_command_buffer(
        StreamTag stream_tag) noexcept { return nullptr; }
    virtual bool execute_command_buffer(VkCommandBuffer cmd_buffer) noexcept { return false; }
    virtual void readback_vulkan_device(
        VkInstance instance,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkAllocationCallbacks *alloc_callback,
        VkPipelineCacheHeaderVersionOne const &pso_meta,
        VkQueue graphics_queue,
        VkQueue compute_queue,
        VkQueue copy_queue,
        IDxcCompiler3 *dxc_compiler,
        IDxcLibrary *dxc_library,
        IDxcUtils *dxc_utils) noexcept {}
};
}// namespace luisa::compute