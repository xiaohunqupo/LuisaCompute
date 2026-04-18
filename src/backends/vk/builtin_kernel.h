#pragma once
#include "compute_shader.h"
namespace lc::vk {
class BuiltinKernel {
public:
    static ComputeShader *load_bindless_set_kernel(Device *device);
    static ComputeShader *load_accel_set_kernel(Device *device);
    // static ComputeShader *LoadBC6TryModeG10CSKernel(Device *device);
    // static ComputeShader *LoadBC6TryModeLE10CSKernel(Device *device);
    // static ComputeShader *LoadBC6EncodeBlockCSKernel(Device *device);
    // static ComputeShader *LoadBC7TryMode456CSKernel(Device *device);
    // static ComputeShader *LoadBC7TryMode137CSKernel(Device *device);
    // static ComputeShader *LoadBC7TryMode02CSKernel(Device *device);
    // static ComputeShader *LoadBC7EncodeBlockCSKernel(Device *device);
};
}// namespace lc::vk

