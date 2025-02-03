#pragma once

#include <luisa/core/dll_export.h>

#include <luisa/core/concepts.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>

namespace luisa::compute::xir {

#define LUISA_XIR_DEFINED_ISA_METHOD(BaseType, base_name)                                            \
    template<typename Derived>                                                                       \
        requires std::derived_from<Derived, BaseType>                                                \
    [[nodiscard]] bool isa() const noexcept {                                                        \
        if constexpr (std::is_same_v<BaseType, Derived>) {                                           \
            return true;                                                                             \
        } else {                                                                                     \
            using ImmediateDerived = typename Derived::derived_##base_name##_type;                   \
            if constexpr (std::is_final_v<ImmediateDerived>) {                                       \
                return derived_##base_name##_tag() == Derived::static_derived_##base_name##_tag();   \
            } else {                                                                                 \
                return derived_##base_name##_tag() == Derived::static_derived_##base_name##_tag() && \
                       static_cast<const ImmediateDerived *>(this)->template isa<Derived>();         \
            }                                                                                        \
        }                                                                                            \
    }

class Pool;

class LC_XIR_API PooledObject {

protected:
    explicit PooledObject() noexcept = default;

public:
    virtual ~PooledObject() noexcept = default;
    [[nodiscard]] virtual Pool *pool() const noexcept = 0;

    // make the object pinned to its memory location
    PooledObject(PooledObject &&) noexcept = delete;
    PooledObject(const PooledObject &) noexcept = delete;
    PooledObject &operator=(PooledObject &&) noexcept = delete;
    PooledObject &operator=(const PooledObject &) noexcept = delete;
};

class LC_XIR_API Pool : public concepts::Noncopyable {

private:
    luisa::vector<PooledObject *> _objects;

public:
    explicit Pool(size_t init_cap = 0u) noexcept;
    ~Pool() noexcept;

public:
    template<typename T, typename... Args>
        requires std::derived_from<T, PooledObject>
    [[nodiscard]] T *create(Args &&...args) {
        auto object = luisa::new_with_allocator<T>(std::forward<Args>(args)...);
        _objects.emplace_back(object);
        return object;
    }
};

class LC_XIR_API PoolOwner {

private:
    luisa::unique_ptr<Pool> _pool;

public:
    explicit PoolOwner(size_t init_pool_cap = 0u) noexcept;
    virtual ~PoolOwner() noexcept = default;
    [[nodiscard]] Pool *pool() const noexcept { return _pool.get(); }
};

}// namespace luisa::compute::xir
