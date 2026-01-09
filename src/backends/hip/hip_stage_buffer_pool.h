//
// Created by mike on 1/9/26.
//

#pragma once

#include <luisa/core/spin_mutex.h>
#include <luisa/core/first_fit.h>
#include "hip_callback_context.h"

namespace luisa::compute::hip {

class HIPStageBufferPool {

public:
    static constexpr auto alignment = 16u;

public:
    class View final : public HIPCallbackContext {

    private:
        void *_handle;
        HIPStageBufferPool *_pool{nullptr};

    public:
        explicit View(std::byte *handle) noexcept;
        View(FirstFit::Node *node, HIPStageBufferPool *pool) noexcept;
        [[nodiscard]] auto is_pooled() const noexcept { return _pool != nullptr; }
        [[nodiscard]] auto node() const noexcept { return static_cast<FirstFit::Node *>(_handle); }
        [[nodiscard]] std::byte *address() const noexcept;
        [[nodiscard]] static View *create(std::byte *handle) noexcept;
        [[nodiscard]] static View *create(FirstFit::Node *node, HIPStageBufferPool *pool) noexcept;
        void recycle() noexcept override;
    };

private:
    spin_mutex _mutex;
    std::byte *_memory{nullptr};
    FirstFit _first_fit;

public:
    HIPStageBufferPool(size_t size, bool write_combined) noexcept;
    ~HIPStageBufferPool() noexcept;
    [[nodiscard]] std::byte *memory() const noexcept { return _memory; }
    [[nodiscard]] View *allocate(size_t size, bool fallback_if_failed = true) noexcept;
    void recycle(FirstFit::Node *node) noexcept;
};

}// namespace luisa::compute::hip
