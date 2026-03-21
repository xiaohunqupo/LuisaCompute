#include <mutex>

#include <luisa/core/logging.h>
#include <luisa/runtime/bindless_array.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"
#include "hip_bindless_array.h"
#include "hip_texture.h"

namespace luisa::compute::hip {

HIPBindlessArray::HIPBindlessArray(size_t capacity) noexcept
    : _capacity{capacity},
      _host_slots(capacity, Slot{}),
      _tex2d_slots(capacity, 0ull),
      _tex3d_slots(capacity, 0ull),
      _texture_tracker{capacity} {
    LUISA_CHECK_HIP(hipMalloc(&_handle, capacity * sizeof(Slot)));
    LUISA_CHECK_HIP(hipMemset(_handle, 0, capacity * sizeof(Slot)));
}

HIPBindlessArray::~HIPBindlessArray() noexcept {
    if (_handle) {
        LUISA_CHECK_HIP(hipFree(_handle));
    }
    _texture_tracker.traverse([](auto tex) noexcept {
        auto tex_obj = reinterpret_cast<hipTextureObject_t>(tex);
        LUISA_CHECK_HIP(hipTexObjectDestroy(tex_obj));
    });
}

void HIPBindlessArray::update(HIPCommandEncoder &encoder,
                              BindlessArrayUpdateCommand *cmd) noexcept {

    std::scoped_lock lock{_mutex};

    using Mod = BindlessArrayUpdateCommand::Modification;
    using BufferMod = BindlessArrayUpdateCommand::BufferModification;
    using Tex2DMod = BindlessArrayUpdateCommand::Texture2DModification;
    using Tex3DMod = BindlessArrayUpdateCommand::Texture3DModification;
    using Op = BindlessArrayUpdateCommand::Operation;

    auto stream = encoder.stream()->handle();

    luisa::vector<size_t> dirty_slots;

    auto process_buffer = [&](size_t slot, const auto &buf) noexcept {
        if (buf.op == Op::EMPLACE) {
            auto buffer = reinterpret_cast<const HIPBuffer *>(buf.handle);
            LUISA_ASSERT(buf.offset_bytes < buffer->size_bytes(),
                         "Offset {} exceeds buffer size {}.",
                         buf.offset_bytes, buffer->size_bytes());
            auto address = reinterpret_cast<uint64_t>(buffer->handle()) + buf.offset_bytes;
            auto size = buffer->size_bytes() - buf.offset_bytes;
            _host_slots[slot].buffer = address;
            _host_slots[slot].size = size;
            dirty_slots.emplace_back(slot);
        } else if (buf.op == Op::REMOVE) {
            _host_slots[slot].buffer = 0u;
            _host_slots[slot].size = 0u;
            dirty_slots.emplace_back(slot);
        }
    };

    auto process_tex2d = [&](size_t slot, const auto &tex) noexcept {
        if (tex.op == Op::EMPLACE) {
            if (auto t = _tex2d_slots[slot]) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(t));
            }
            auto texture = reinterpret_cast<const HIPTexture *>(tex.handle);
            auto tex_object = texture->create_texture_object(tex.sampler);
            _tex2d_slots[slot] = tex_object;
            _host_slots[slot].tex2d = reinterpret_cast<uint64_t>(tex_object);
            _texture_tracker.retain(reinterpret_cast<uint64_t>(tex_object));
            dirty_slots.emplace_back(slot);
        } else if (tex.op == Op::REMOVE) {
            if (auto t = _tex2d_slots[slot]) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(t));
            }
            _tex2d_slots[slot] = 0ull;
            _host_slots[slot].tex2d = 0u;
            dirty_slots.emplace_back(slot);
        }
    };

    auto process_tex3d = [&](size_t slot, const auto &tex) noexcept {
        if (tex.op == Op::EMPLACE) {
            if (auto t = _tex3d_slots[slot]) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(t));
            }
            auto texture = reinterpret_cast<const HIPTexture *>(tex.handle);
            auto tex_object = texture->create_texture_object(tex.sampler);
            _tex3d_slots[slot] = tex_object;
            _host_slots[slot].tex3d = reinterpret_cast<uint64_t>(tex_object);
            _texture_tracker.retain(reinterpret_cast<uint64_t>(tex_object));
            dirty_slots.emplace_back(slot);
        } else if (tex.op == Op::REMOVE) {
            if (auto t = _tex3d_slots[slot]) {
                _texture_tracker.release(reinterpret_cast<uint64_t>(t));
            }
            _tex3d_slots[slot] = 0ull;
            _host_slots[slot].tex3d = 0u;
            dirty_slots.emplace_back(slot);
        }
    };

    cmd->visit_modifications([&](auto &mods) noexcept {
        using T = std::decay_t<decltype(mods)>;
        if constexpr (std::is_same_v<T, luisa::vector<Mod>>) {
            for (auto &m : mods) {
                process_buffer(m.slot, m.buffer);
                process_tex2d(m.slot, m.tex2d);
                process_tex3d(m.slot, m.tex3d);
            }
        } else if constexpr (std::is_same_v<T, luisa::vector<BufferMod>>) {
            for (auto &m : mods) {
                process_buffer(m.slot, m.buffer);
            }
        } else if constexpr (std::is_same_v<T, luisa::vector<Tex2DMod>>) {
            for (auto &m : mods) {
                process_tex2d(m.slot, m.tex2d);
            }
        } else if constexpr (std::is_same_v<T, luisa::vector<Tex3DMod>>) {
            for (auto &m : mods) {
                process_tex3d(m.slot, m.tex3d);
            }
        }
    });

    _texture_tracker.commit([](auto tex) noexcept {
        auto tex_obj = reinterpret_cast<hipTextureObject_t>(tex);
        LUISA_CHECK_HIP(hipTexObjectDestroy(tex_obj));
    });

    if (dirty_slots.empty()) { return; }

    std::sort(dirty_slots.begin(), dirty_slots.end());
    dirty_slots.erase(std::unique(dirty_slots.begin(), dirty_slots.end()),
                      dirty_slots.end());

    for (auto slot : dirty_slots) {
        auto dst = static_cast<std::byte *>(_handle) + slot * sizeof(Slot);
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            dst, &_host_slots[slot], sizeof(Slot), stream));
    }
}

void HIPBindlessArray::set_name(luisa::string &&name) noexcept {
    std::scoped_lock lock{_mutex};
    _name = std::move(name);
}

}// namespace luisa::compute::hip
