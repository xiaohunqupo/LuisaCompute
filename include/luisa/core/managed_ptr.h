#pragma once

#include <atomic>
#include <luisa/core/stl/memory.h>

namespace luisa {

namespace detail {

struct ManagedObjectLowLevelOp;

class LC_CORE_API ManagedObject {

private:
    std::atomic<uint32_t> _ref_count;
    uint32_t _managed_id;

public:
    explicit ManagedObject(uint32_t m_id = 0u) noexcept : _ref_count{1u}, _managed_id{m_id} {}
    virtual ~ManagedObject() = default;
    ManagedObject(ManagedObject &&) = delete;
    ManagedObject(const ManagedObject &) = delete;
    ManagedObject &operator=(ManagedObject &&) = delete;
    ManagedObject &operator=(const ManagedObject &) = delete;

private:
    friend ManagedObjectLowLevelOp;
    ManagedObject *do_retain() noexcept;
    void do_release() noexcept;

public:
    [[nodiscard]] auto managed_id() const noexcept { return _managed_id; }
    void set_managed_id(uint32_t m_id) noexcept { _managed_id = m_id; }
};

struct ManagedObjectLowLevelOp {
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto retain_nonnull(T *o) noexcept {
        return static_cast<T *>(o->do_retain());
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    [[nodiscard]] static auto retain(T *o) noexcept {
        return o == nullptr ? nullptr : retain_nonnull(o);
    }
    template<typename T>
        requires std::derived_from<T, ManagedObject>
    static void release(T *o) noexcept {
        if (o != nullptr) { o->do_release(); }
    }
};

}// namespace detail

template<typename T>
class ManagedPtr : public ManagedPtr<const T> {
public:
    ManagedPtr() noexcept = default;
    ~ManagedPtr() noexcept = default;

    ManagedPtr(const ManagedPtr &) noexcept = default;
    ManagedPtr(ManagedPtr &&) noexcept = default;
    ManagedPtr &operator=(ManagedPtr &&) noexcept = default;
    ManagedPtr &operator=(const ManagedPtr &) noexcept = default;

    ManagedPtr(const ManagedPtr<const T> &) noexcept = delete;
    ManagedPtr(ManagedPtr<const T> &&) noexcept = delete;
    ManagedPtr &operator=(ManagedPtr<const T> &&) noexcept = delete;
    ManagedPtr &operator=(const ManagedPtr<const T> &) noexcept = delete;

    [[nodiscard]] T *get() const noexcept { return const_cast<T *>(ManagedPtr<const T>::get()); }
    [[nodiscard]] T *operator->() const noexcept { return get(); }
    [[nodiscard]] T &operator*() const noexcept { return *get(); }
};

template<typename T>
class ManagedPtr<const T> {

    static_assert(std::derived_from<T, detail::ManagedObject>);

private:
    T *_object{nullptr};
    [[nodiscard]] T *_transfer_ownership() noexcept {
        return std::exchange(_object, nullptr);
    }

public:
    ManagedPtr() noexcept = default;
    ~ManagedPtr() noexcept { reset(); }
    ManagedPtr(ManagedPtr &&other) noexcept {
        reset(other._transfer_ownership());
    }
    ManagedPtr(const ManagedPtr &other) noexcept {
        reset(luisa::detail::ManagedObjectLowLevelOp::
                  retain(const_cast<T *>(other.get())));
    }
    ManagedPtr &operator=(ManagedPtr &&other) noexcept {
        if (&other != this) { reset(other._transfer_ownership()); }
        return *this;
    }
    ManagedPtr &operator=(const ManagedPtr &other) noexcept {
        if (&other != this) {
            reset(luisa::detail::ManagedObjectLowLevelOp::
                      retain(const_cast<T *>(other.get())));
        }
        return *this;
    }

public:
    [[nodiscard]] const T *get() const noexcept { return _object; }
    [[nodiscard]] const T *operator->() const noexcept { return get(); }
    [[nodiscard]] const T &operator*() const noexcept { return *get(); }
    [[nodiscard]] explicit operator bool() const noexcept { return _object != nullptr; }

public:
    void reset(const T *new_object = nullptr) noexcept {
        luisa::detail::ManagedObjectLowLevelOp::release(
            std::exchange(_object, const_cast<T *>(new_object)));
    }
};

template<typename T, typename Base = detail::ManagedObject>
    requires std::derived_from<Base, detail::ManagedObject>
class Managed : public Base {

    static_assert(std::same_as<std::remove_cv_t<T>, T>);

public:
    using derived_type = T;
    using base_type = Base;
    using super_type = Managed;
    using base_type::base_type;

public:
    [[nodiscard]] auto lock() noexcept {
        ManagedPtr<T> p;
        p.reset(detail::ManagedObjectLowLevelOp::
                    retain_nonnull(static_cast<T *>(this)));
        return p;
    }
    [[nodiscard]] auto lock() const noexcept {
        return static_cast<ManagedPtr<const T>>(
            const_cast<Managed *>(this)->lock());
    }
};

template<typename T, typename... Args>
    requires std::derived_from<T, detail::ManagedObject>
[[nodiscard]] ManagedPtr<T> make_managed(Args &&...args) noexcept {
    auto o = luisa::new_with_allocator<std::remove_const_t<T>>(std::forward<Args>(args)...);
    assert(std::addressof(*o) == std::addressof(*static_cast<detail::ManagedObject *>(o)) &&
           "ManagedObject should be the first non-empty base class of its derived classes.");
    ManagedPtr<T> ret;
    ret.reset(o);
    return ret;
}

}// namespace luisa
