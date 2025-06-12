#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/ast/type.h>
namespace luisa::clangcxx {
struct BuildArgument {
    compute::Type const *type;
    string var_name;
};
}// namespace luisa::clangcxx