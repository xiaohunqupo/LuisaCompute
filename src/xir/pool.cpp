#include <luisa/core/logging.h>
#include <luisa/xir/pool.h>

namespace luisa::compute::xir {

Pool::Pool(size_t init_cap) noexcept {
    if (init_cap != 0u) {
        _objects.reserve(init_cap);
    }
}

Pool::~Pool() noexcept {
    for (auto object : _objects) {
        LUISA_DEBUG_ASSERT(object->pool() == this, "Detected object from another pool.");
        luisa::delete_with_allocator(object);
    }
}

PoolOwner::PoolOwner(size_t init_pool_cap) noexcept
    : _pool{luisa::make_unique<Pool>(init_pool_cap)} {}

}// namespace luisa::compute::xir
