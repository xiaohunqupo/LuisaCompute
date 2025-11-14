#pragma once
#include <luisa/backends/ext/pinned_memory_ext.hpp>
#include <luisa/vstl/common.h>
namespace lc::validation {
using namespace luisa::compute;

class PinnedMemoryExtImpl : public PinnedMemoryExt, public vstd::IOperatorNewBase {
    PinnedMemoryExt *_impl;
protected:
    [[nodiscard]] BufferCreationInfo _pin_host_memory(
        const Type *elem_type, size_t elem_count,
        void *host_ptr, const PinnedMemoryOption &option) noexcept override;

    [[nodiscard]] BufferCreationInfo _allocate_pinned_memory(
        const Type *elem_type, size_t elem_count,
        const PinnedMemoryOption &option) noexcept override;
public:
    explicit PinnedMemoryExtImpl(PinnedMemoryExt *impl) : _impl(impl) {}
    [[nodiscard]] DeviceInterface *device() const noexcept override {
        return _impl->device();
    }
    void flush_range(
        uint64_t buffer_handle,
        uint64_t begin,
        uint64_t end) const noexcept override;
};
}// namespace lc::validation