#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>

namespace luisa {

LUISA_EXPORT_API void *allocator_allocate(size_t size, size_t alignment) noexcept {
#ifdef LUISA_USE_SYSTEM_STL
    return ::aligned_alloc(alignment, size);
#else
    return eastl::GetDefaultAllocator()->allocate(size, alignment, 0u);
#endif
}

LUISA_EXPORT_API void allocator_deallocate(void *p, size_t) noexcept {
#ifdef LUISA_USE_SYSTEM_STL
    return ::free(p);
#else
    eastl::GetDefaultAllocator()->deallocate(p, 0u);
#endif
}

LUISA_EXPORT_API void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
#ifdef LUISA_USE_SYSTEM_STL
    p = ::realloc(p, size);
#else
    p = eastl::GetDefaultAllocator()->reallocate(p, size);
#endif
    LUISA_DEBUG_ASSERT(alignment == 0u || reinterpret_cast<uintptr_t>(p) % alignment == 0u,
                       "Reallocation failed to maintain alignment.");
    return p;
}

}// namespace luisa
