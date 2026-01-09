//
// Created by mike on 1/9/26.
//

#include <luisa/core/pool.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_command_encoder.h"

#include "luisa/core/logging.h"

namespace luisa::compute::hip {

class UserCallbackContext : public HIPCallbackContext {

public:
    using CallbackContainer = CommandList::CallbackContainer;

private:
    CallbackContainer _functions;

    [[nodiscard]] static auto &_object_pool() noexcept {
        static Pool<UserCallbackContext, true> pool;
        return pool;
    }

public:
    explicit UserCallbackContext(CallbackContainer &&cbs) noexcept
        : _functions{std::move(cbs)} {}

    [[nodiscard]] static auto create(CallbackContainer &&cbs) noexcept {
        return _object_pool().create(std::move(cbs));
    }

    void recycle() noexcept override {
        for (auto &&f : _functions) { f(); }
        _object_pool().destroy(this);
    }
};

void HIPCommandEncoder::commit(CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        _callbacks.emplace_back(
            UserCallbackContext::create(
                std::move(user_callbacks)));
    }
    if (auto callbacks = std::move(_callbacks); !callbacks.empty()) {
        _stream->callback(std::move(callbacks));
    }
}

void HIPCommandEncoder::visit(BufferUploadCommand *command) noexcept {
    LUISA_DEBUG_ASSERT(command->handle() != invalid_resource_handle);
    auto buffer = reinterpret_cast<const HIPBuffer *>(command->handle());
    auto address = static_cast<std::byte *>(buffer->handle()) + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_upload_buffer(size, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size);
        LUISA_CHECK_HIP(hipMemcpyHtoDAsync(
            address, upload_buffer->address(),
            size, _stream->handle()));
    });
}

void HIPCommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    LUISA_DEBUG_ASSERT(command->handle() != invalid_resource_handle);
    auto buffer = reinterpret_cast<const HIPBuffer *>(command->handle());
    auto address = static_cast<std::byte *>(buffer->handle()) + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_download_pool_no_fallback(size, [&](auto download_buffer) noexcept {
        if (download_buffer) {
            LUISA_CHECK_HIP(hipMemcpyDtoHAsync(
                download_buffer->address(), address,
                size, _stream->handle()));
            LUISA_CHECK_HIP(hipMemcpyAsync(
                data, download_buffer->address(),
                size, hipMemcpyHostToHost, _stream->handle()));
        } else {
            LUISA_CHECK_HIP(hipMemcpyDtoHAsync(
                data, address, size, _stream->handle()));
        }
    });
}

void HIPCommandEncoder::visit(BufferCopyCommand *command) noexcept {
    LUISA_DEBUG_ASSERT(command->src_handle() != invalid_resource_handle);
    LUISA_DEBUG_ASSERT(command->dst_handle() != invalid_resource_handle);
    auto src_buffer = reinterpret_cast<const HIPBuffer *>(command->src_handle());
    auto dst_buffer = reinterpret_cast<const HIPBuffer *>(command->dst_handle());
    auto src_address = static_cast<std::byte *>(src_buffer->handle()) + command->src_offset();
    auto dst_address = static_cast<std::byte *>(dst_buffer->handle()) + command->dst_offset();
    auto size = command->size();
    LUISA_CHECK_HIP(hipMemcpyDtoDAsync(dst_address, src_address, size, _stream->handle()));
}

void HIPCommandEncoder::visit(BufferToTextureCopyCommand *) noexcept {
}

void HIPCommandEncoder::visit(ShaderDispatchCommand *) noexcept {
}

void HIPCommandEncoder::visit(TextureUploadCommand *) noexcept {
}

void HIPCommandEncoder::visit(TextureDownloadCommand *) noexcept {
}

void HIPCommandEncoder::visit(TextureCopyCommand *) noexcept {
}

void HIPCommandEncoder::visit(TextureToBufferCopyCommand *) noexcept {
}

void HIPCommandEncoder::visit(AccelBuildCommand *) noexcept {
}

void HIPCommandEncoder::visit(MeshBuildCommand *) noexcept {
}

void HIPCommandEncoder::visit(CurveBuildCommand *) noexcept {
}

void HIPCommandEncoder::visit(ProceduralPrimitiveBuildCommand *) noexcept {
}

void HIPCommandEncoder::visit(MotionInstanceBuildCommand *) noexcept {
}

void HIPCommandEncoder::visit(BindlessArrayUpdateCommand *) noexcept {
}

void HIPCommandEncoder::visit(CustomCommand *) noexcept {
}

}// namespace luisa::compute::hip
