#pragma once
#include <luisa/runtime/device.h>
#include <luisa/tensor/expression.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/vstl/md5.h>
#include <luisa/vstl/hash_map.h>
namespace luisa::compute {
class LUISA_TENSOR_API ShaderManager {
public:
    struct Key {
        TensorExpr::Tag tag;
        std::array<uint, 4> user_ids;
    };
    struct KeyHashEqual {
        size_t operator()(Key const &k) const noexcept {
            return luisa::hash64(&k, sizeof(Key), luisa::hash64_default_seed);
        }
        int operator()(Key const &a, Key const &b) const noexcept {
            return std::memcmp(&a, &b, sizeof(Key));
        }
    };
    struct ShaderDispatch {
        uint64_t shader_handle;
        uint3 desired_dispatch_size;
        size_t uniform_size;
    };
private:
    vstd::HashMap<Key, std::pair<luisa::spin_mutex, ShaderDispatch>, KeyHashEqual, KeyHashEqual> _shaders;
    luisa::spin_mutex global_mtx;
    DeviceInterface *_device;
public:
    ShaderManager(DeviceInterface *device) noexcept;
    template<typename Lambda>
        requires(std::is_invocable_r_v<ShaderDispatch, Lambda>)
    ShaderDispatch const &add_shader(
        TensorExpr::Tag tag,
        vstd::MD5 hash,
        Lambda &&lambda) noexcept {
        std::array<uint, 4> arr;
        static_assert(sizeof(arr) == sizeof(vstd::MD5));
        std::memcpy(arr.data(), &hash, sizeof(vstd::MD5));
        global_mtx.lock();
        auto iter = _shaders.try_emplace(Key{tag, arr});
        auto &v = iter.first;
        global_mtx.unlock();
        std::lock_guard lck{v.value().first};
        if (iter.second) {
            v.value().second = lambda();
            LUISA_ASSERT(v.value().second.shader_handle != invalid_resource_handle);
        }
        return v.value().second;
    }
    ~ShaderManager() noexcept;
};
}// namespace luisa::compute