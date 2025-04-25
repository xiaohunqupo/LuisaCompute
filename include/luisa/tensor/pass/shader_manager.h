#pragma once
#include <luisa/runtime/device.h>
#include <luisa/tensor/expression.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/vstl/md5.h>
namespace luisa::compute {
class LC_TENSOR_API ShaderManager {
public:
    struct Key {
        TensorExpr::Tag tag;
        std::array<uint, 4> user_ids;
    };
    struct KeyHashEqual {
        size_t operator()(Key const& k) const noexcept {
            return luisa::hash64(&k, sizeof(Key), luisa::hash64_default_seed);
        }
        bool operator()(Key const& a, Key const& b) const noexcept {
            return std::memcmp(&a, &b, sizeof(Key)) == 0;
        }
    };
    struct ShaderDispatch {
        uint64_t shader_handle;
        uint3 desired_dispatch_size;
        size_t uniform_size;
    };
private:
    luisa::unordered_map<Key, ShaderDispatch, KeyHashEqual, KeyHashEqual> _shaders;
    DeviceInterface *_device;
public:
    ShaderManager(DeviceInterface *device) noexcept;
    template<typename Lambda>
        requires(std::is_invocable_r_v<ShaderDispatch, Lambda>)
    ShaderDispatch add_shader(
        TensorExpr::Tag tag,
        vstd::MD5 hash,
        Lambda &&lambda) noexcept {
        std::array<uint, 4> arr;
        static_assert(sizeof(arr) == sizeof(vstd::MD5));
        std::memcpy(arr.data(), &hash, sizeof(vstd::MD5));
        auto iter = _shaders.try_emplace(Key{tag, arr}, luisa::lazy_construct([&]() -> ShaderDispatch {
                                             auto handle = lambda();
                                             LUISA_ASSERT(handle.shader_handle != invalid_resource_handle);
                                             return handle;
                                         }));
        return iter.first->second;
    }
    ~ShaderManager() noexcept;
};
}// namespace luisa::compute