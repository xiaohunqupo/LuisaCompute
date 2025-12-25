//
// Created by mike on 12/25/25.
//

#include <luisa/core/dll_export.h>

#include "hip_check.h"
#include "hip_device.h"

namespace luisa::compute::hip {

struct HIPDeviceGuard {
    int current_device_id = 0;
    int previous_device_id = 0;
    explicit HIPDeviceGuard(int device_id) noexcept : current_device_id{device_id} {
        LUISA_CHECK_HIP(hipGetDevice(&previous_device_id));
        if (previous_device_id != current_device_id) {
            LUISA_CHECK_HIP(hipSetDevice(current_device_id));
        }
    }
    ~HIPDeviceGuard() noexcept {
        if (current_device_id != previous_device_id) {
            LUISA_CHECK_HIP(hipSetDevice(previous_device_id));
        }
    }
    HIPDeviceGuard(HIPDeviceGuard &&) noexcept = delete;
    HIPDeviceGuard(const HIPDeviceGuard &) noexcept = delete;
    HIPDeviceGuard &operator=(HIPDeviceGuard &&) noexcept = delete;
    HIPDeviceGuard &operator=(const HIPDeviceGuard &) noexcept = delete;
};

template<typename F>
decltype(auto) HIPDevice::with_device(F &&f) const noexcept {
    HIPDeviceGuard guard{_device_id};
    return std::invoke(std::forward<F>(f));
}

HIPDevice::HIPDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)},
      _default_io{config == nullptr || config->binary_io == nullptr ?
                      luisa::make_unique<DefaultBinaryIO>(context()) :
                      nullptr},
      _io{_default_io == nullptr ? config->binary_io : _default_io.get()},
      _device_id{config == nullptr || config->device_index == ~0ull ? 0 : static_cast<int>(config->device_index)} {
    auto device_count = 0;
    LUISA_CHECK_HIP(hipGetDeviceCount(&device_count));
    LUISA_ASSERT(_device_id < device_count,
                 "HIP device index out of range (required = {}, count = {}).",
                 _device_id, device_count);
    // log device name
    hipDeviceProp_t prop;
    LUISA_CHECK_HIP(hipGetDeviceProperties(&prop, _device_id));
    LUISA_INFO("Created HIP device {}: {}.", _device_id, prop.name);
}

HIPDevice::~HIPDevice() noexcept = default;

void HIPDevice::set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept {
    DeviceInterface::set_stream_log_callback(stream_handle, callback);
}

ShaderCreationInfo HIPDevice::create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::create_curve(const AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_curve(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::create_motion_instance(const AccelMotionOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_motion_instance(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

luisa::string HIPDevice::query(luisa::string_view property) noexcept {
    // TODO: support more properties
    return DeviceInterface::query(property);
}

DeviceExtension *HIPDevice::extension(luisa::string_view name) noexcept {
    // TODO: support extensions
    return DeviceInterface::extension(name);
}

luisa::string_view HIPDevice::get_name(uint64_t resource_handle) const noexcept {
    return DeviceInterface::get_name(resource_handle);
}

SparseBufferCreationInfo HIPDevice::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::update_sparse_resources(uint64_t stream_handle, luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_sparse_buffer(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

SparseTextureCreationInfo HIPDevice::create_sparse_texture(PixelFormat format, uint dimension,
                                                           uint width, uint height, uint depth,
                                                           uint mipmap_levels, bool simultaneous_access) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_sparse_texture(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void *HIPDevice::native_handle() const noexcept {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(_device_id));
}

uint HIPDevice::compute_warp_size() const noexcept {
    int warp_size = 0;
    LUISA_CHECK_HIP(hipDeviceGetAttribute(
        &warp_size,
        hipDeviceAttributeWarpSize,
        _device_id));
    return static_cast<uint>(warp_size);
}

uint64_t HIPDevice::memory_granularity() const noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::BufferCreationInfo HIPDevice::create_buffer(const compute::Type *element, size_t elem_count, void *external_memory) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::BufferCreationInfo HIPDevice::create_buffer(const compute::ir::CArc<compute::ir::Type> *element, size_t elem_count, void *external_memory) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_buffer(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ResourceCreationInfo HIPDevice::create_texture(compute::PixelFormat format, uint dimension,
                                                        uint width, uint height, uint depth,
                                                        uint mipmap_levels, void *external_native_handle,
                                                        bool simultaneous_access, bool allow_raster_target) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_texture(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ResourceCreationInfo HIPDevice::create_bindless_array(size_t size, compute::BindlessSlotType type) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_bindless_array(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ResourceCreationInfo HIPDevice::create_stream(compute::StreamTag stream_tag) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_stream(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::dispatch(uint64_t stream_handle, compute::CommandList &&list) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::SwapchainCreationInfo HIPDevice::create_swapchain(const compute::SwapchainOption &option, uint64_t stream_handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_swap_chain(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ShaderCreationInfo HIPDevice::create_shader(const compute::ShaderOption &option, compute::Function kernel) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ShaderCreationInfo HIPDevice::create_shader(const compute::ShaderOption &option, const compute::ir::KernelModule *kernel) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ShaderCreationInfo HIPDevice::load_shader(luisa::string_view name, luisa::span<compute::Type const *const> arg_types) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::Usage HIPDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_shader(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ResourceCreationInfo HIPDevice::create_event() noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_event(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

bool HIPDevice::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ResourceCreationInfo HIPDevice::create_mesh(const compute::AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_mesh(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ResourceCreationInfo HIPDevice::create_procedural_primitive(const compute::AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

compute::ResourceCreationInfo HIPDevice::create_accel(const compute::AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_accel(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::set_name(luisa::compute::Resource::Tag resource_tag,
                         uint64_t resource_handle,
                         luisa::string_view name) noexcept {
    // ignored
}

}// namespace luisa::compute::hip

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::new_with_allocator<luisa::compute::hip::HIPDevice>(std::move(ctx), config);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    auto count = 0;
    LUISA_CHECK_HIP(hipGetDeviceCount(&count));
    names.reserve(count);
    for (int i = 0; i < count; i++) {
        hipDeviceProp_t prop;
        LUISA_CHECK_HIP(hipGetDeviceProperties(&prop, i));
        names.emplace_back(prop.name);
    }
}

#include "../common/export_version.inl.h"
