#pragma once
#include <luisa/ast/function.h>
namespace lc::hlsl {
using namespace luisa;
using namespace luisa::compute;
struct SharedVar {
    Function func;
    Variable var;
    bool operator==(SharedVar const &v) const {
        return v.func == func && v.var.uid() == var.uid();
    }
};
struct SharedVarHash {
    size_t operator()(SharedVar const &s) const {
        return hash_combine({s.func.hash(), s.var.hash()});
    }
};
using SharedVarSet = luisa::unordered_set<SharedVar, SharedVarHash>;
}// namespace lc::hlsl