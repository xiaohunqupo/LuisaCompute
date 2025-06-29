#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/core/logging.h>

namespace luisa::compute {

namespace detail {

void ShaderInvokeBase::encode(ShaderDispatchCmdEncoder &encoder, const BindlessArray &array) noexcept {
    array._check_is_valid();
#ifndef NDEBUG
    if (array.dirty()) [[unlikely]] {
        LUISA_WARNING("Dispatching shader with a dirty bindless array.");
    }
#endif
    encoder.encode_bindless_array(array.handle());
}

}// namespace detail

BindlessArray Device::create_bindless_array(size_t slots, BindlessType type) noexcept {
    return _create<BindlessArray>(slots, type);
}

BindlessArray::BindlessArray(BindlessArray &&rhs) noexcept
    : Resource{std::move(rhs)},
      _size{rhs._size},
      _updates{std::move(rhs._updates)} {
    rhs._size = 0;
}

BindlessArray::BindlessArray(DeviceInterface *device, size_t size, BindlessType type) noexcept
    : Resource{device, Tag::BINDLESS_ARRAY, device->create_bindless_array(size, type)},
      _size{size} {
    switch (type) {
        case BindlessType::None:
            _updates.emplace<luisa::unordered_set<Modification, ModSlotHash, ModSlotEqual>>();
            break;
        case BindlessType::Buffer:
            _updates.emplace<luisa::unordered_set<BufferModification, ModSlotHash, ModSlotEqual>>();
            break;
        case BindlessType::Texture2D:
            _updates.emplace<luisa::unordered_set<Texture2DModification, ModSlotHash, ModSlotEqual>>();
            break;
        case BindlessType::Texture3D:
            _updates.emplace<luisa::unordered_set<Texture3DModification, ModSlotHash, ModSlotEqual>>();
            break;
    }
}

void BindlessArray::emplace_buffer_handle_on_update(size_t index, uint64_t handle, size_t offset_bytes) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    if (_updates.index() == 0) {
        auto [iter, _] = luisa::get<0>(_updates).emplace(index);
        const_cast<Modification::Buffer &>(iter->buffer) = Modification::Buffer::emplace(handle, offset_bytes);
    } else if (_updates.index() == 1) {
        auto [iter, _] = luisa::get<1>(_updates).emplace(index);
        const_cast<Modification::Buffer &>(iter->buffer) = Modification::Buffer::emplace(handle, offset_bytes);
    } else {
        LUISA_ASSERT(false, "Invalid bindless type.");
    }
}

void BindlessArray::emplace_tex2d_handle_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    if (_updates.index() == 0) {
        auto [iter, _] = luisa::get<0>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex2d) = Modification::Texture::emplace(handle, sampler);
    } else if (_updates.index() == 2) {
        auto [iter, _] = luisa::get<2>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex2d) = Modification::Texture::emplace(handle, sampler);
    } else {
        LUISA_ASSERT(false, "Invalid bindless type.");
    }
}

void BindlessArray::emplace_tex3d_handle_on_update(size_t index, uint64_t handle, Sampler sampler) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    if (_updates.index() == 0) {
        auto [iter, _] = luisa::get<0>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex3d) = Modification::Texture::emplace(handle, sampler);
    } else if (_updates.index() == 3) {
        auto [iter, _] = luisa::get<3>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex3d) = Modification::Texture::emplace(handle, sampler);
    } else {
        LUISA_ASSERT(false, "Invalid bindless type.");
    }
}

BindlessArray &BindlessArray::remove_buffer_on_update(size_t index) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid buffer slot {} for bindless array of size {}.",
            index, _size);
    }
    if (_updates.index() == 0) {
        auto [iter, _] = luisa::get<0>(_updates).emplace(index);
        const_cast<Modification::Buffer &>(iter->buffer) = Modification::Buffer::remove();
    } else if (_updates.index() == 1) {
        auto [iter, _] = luisa::get<1>(_updates).emplace(index);
        const_cast<Modification::Buffer &>(iter->buffer) = Modification::Buffer::remove();
    } else {
        LUISA_ASSERT(false, "Invalid bindless type.");
    }
    return *this;
}

BindlessArray &BindlessArray::remove_tex2d_on_update(size_t index) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture2d slot {} for bindless array of size {}.",
            index, _size);
    }
    if (_updates.index() == 0) {
        auto [iter, _] = luisa::get<0>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex2d) = Modification::Texture::remove();
    } else if (_updates.index() == 2) {
        auto [iter, _] = luisa::get<2>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex2d) = Modification::Texture::remove();
    } else {
        LUISA_ASSERT(false, "Invalid bindless type.");
    }
    return *this;
}

BindlessArray &BindlessArray::remove_tex3d_on_update(size_t index) noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (index >= _size) [[unlikely]] {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid texture3d slot {} for bindless array of size {}.",
            index, _size);
    }
    if (_updates.index() == 0) {
        auto [iter, _] = luisa::get<0>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex3d) = Modification::Texture::remove();
    } else if (_updates.index() == 3) {
        auto [iter, _] = luisa::get<3>(_updates).emplace(index);
        const_cast<Modification::Texture &>(iter->tex3d) = Modification::Texture::remove();
    }
    return *this;
}

luisa::unique_ptr<Command> BindlessArray::update() noexcept {
    _check_is_valid();
    std::lock_guard lock{_mtx};
    if (luisa::visit([](auto &&t) { return t.empty(); }, _updates)) {
        LUISA_WARNING_WITH_LOCATION(
            "No update to bindless array.");
        return nullptr;
    }
    using VV = std::remove_cvref_t<decltype(luisa::get<0>(_updates))>::key_type;
    return luisa::visit(
        [this]<typename T>(luisa::unordered_set<T, ModSlotHash, ModSlotEqual> &updates) {
            luisa::vector<T> mods;
            mods.reserve(updates.size());
            for (auto m : updates) { mods.emplace_back(m); }
            updates.clear();
            return luisa::make_unique<BindlessArrayUpdateCommand>(handle(), std::move(mods));
        },
        _updates);
}

BindlessArray::~BindlessArray() noexcept {
    if (!luisa::visit([](auto &&t) { return t.empty(); }, _updates)) {
        LUISA_WARNING_WITH_LOCATION(
            "Bindless array #{} destroyed with {} pending updates. "
            "Did you forget to call update()?",
            this->handle(),
            luisa::visit([](auto &&t) { return t.size(); }, _updates));
    }
    if (*this) {
        device()->destroy_bindless_array(handle());
    }
}

}// namespace luisa::compute
