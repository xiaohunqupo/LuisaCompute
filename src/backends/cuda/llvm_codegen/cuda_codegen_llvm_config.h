//
// Created by mike on 9/19/25.
//

#pragma once

#include <luisa/core/stl/memory.h>

namespace luisa::compute::cuda {

struct CUDACodegenLLVMConfig {
    luisa::span<const std::byte> libdevice_bitcode;
    bool enable_fast_math{true};
    bool enable_debug_info{false};
};

}// namespace luisa::compute::cuda
