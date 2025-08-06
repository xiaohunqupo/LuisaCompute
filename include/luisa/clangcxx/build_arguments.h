#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/ast/type.h>
#include <luisa/ast/usage.h>
namespace luisa::clangcxx {
struct BuildArgument {
    compute::Type const *type{};
    string var_name;
    uint resource_var_id{~0u};
    compute::Usage var_usage{compute::Usage::NONE};
};
struct ShaderReflection {
    uint dimension;
    uint3 block_size;
    luisa::vector<BuildArgument> kernel_args;
};
}// namespace luisa::clangcxx