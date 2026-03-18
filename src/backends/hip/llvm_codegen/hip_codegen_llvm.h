//
// Created by mike on 3/18/26.
//

#pragma once

#include <luisa/core/stl/string.h>
#include <luisa/ast/function.h>

namespace luisa::compute::xir {
class Module;
}// namespace luisa::compute::xir

namespace luisa::compute::hip {

struct HIPCodegenLLVMConfig {

    enum struct OptLevel : uint8_t {
        LEVEL_NONE = 0,
        LEVEL_LESS = 1,
        LEVEL_DEFAULT = 2,
        LEVEL_AGGRESSIVE = 3,
    };

    luisa::string source_file{};
    luisa::span<const Function::Binding> bindings{};
    std::array<uint32_t, 3> block_size{};
    uint32_t amdgpu_arch{};
    OptLevel opt_level{OptLevel::LEVEL_AGGRESSIVE};
    bool enable_fast_math{true};
    bool enable_debug_info{false};
};

[[nodiscard]] luisa::string hip_codegen_llvm(
    const xir::Module &xir_module,
    const HIPCodegenLLVMConfig &config) noexcept;

}// namespace luisa::compute::hip
