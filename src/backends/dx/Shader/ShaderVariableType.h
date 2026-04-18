#pragma once
#include <luisa/vstl/common.h>
#include <Windows.h>
#include <d3dx12.h>
#include "../../common/hlsl/shader_property.h"
namespace lc::dx {

enum class HitGroupFunctionType : uint8_t {
    kRayGeneration,
    kClosestHit,
    kAnyHit,
    kMiss,
    kIntersect,
    kNum
};
struct DXRHitGroup {
    D3D12_HIT_GROUP_TYPE shader_type;
    D3D12_GPU_VIRTUAL_ADDRESS miss_v_address;
    uint64 miss_size;
    D3D12_GPU_VIRTUAL_ADDRESS ray_gen_v_address;
    uint64 ray_gen_size;
    D3D12_GPU_VIRTUAL_ADDRESS hit_group_v_address;
    uint64 hit_group_size;
};
struct ShaderVariable {
    vstd::string name;
    hlsl::ShaderVariableType type;
    uint table_size;
    uint register_pos;
    uint space;
    ShaderVariable() = default;
    ShaderVariable(
        const vstd::string &name,
        hlsl::ShaderVariableType type,
        uint table_size,
        uint register_pos,
        uint space)
        : name(name),
          type(type),
          table_size(table_size),
          register_pos(register_pos),
          space(space) {}
};
}// namespace lc::dx
