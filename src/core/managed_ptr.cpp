#include <luisa/core/logging.h>
#include <luisa/core/managed_ptr.h>

namespace luisa {

ManagedObject *ManagedObject::retain() noexcept {
    LUISA_DEBUG_ASSERT(this != nullptr, "Retaining a null object.");
    [[maybe_unused]] auto old_refcount = _ref_count.fetch_add(1, std::memory_order_relaxed);
    LUISA_DEBUG_ASSERT(old_refcount > 0, "Retained object is likely already destroyed.");
    return this;
}

void ManagedObject::release() noexcept {
    LUISA_DEBUG_ASSERT(this != nullptr, "Releasing a null object.");
    auto old_refcount = _ref_count.fetch_sub(1, std::memory_order_acq_rel);
    LUISA_DEBUG_ASSERT(old_refcount > 0, "Releasing object is likely already destroyed.");
    if (old_refcount == 1) {
        luisa::delete_with_allocator(this);
    }
}

}// namespace luisa
