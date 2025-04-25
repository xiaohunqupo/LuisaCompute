#include <luisa/tensor/pass/shader_manager.h>
namespace luisa::compute {
ShaderManager::ShaderManager(DeviceInterface *device) noexcept
    : _device(device) {}
ShaderManager::~ShaderManager() noexcept {
    for (auto &i : _shaders) {
        _device->destroy_shader(i.second.shader_handle);
    }
}

}// namespace luisa::compute