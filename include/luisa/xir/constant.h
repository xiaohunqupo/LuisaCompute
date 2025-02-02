#pragma once

#include <luisa/xir/value.h>

namespace luisa::compute::xir {

class LC_XIR_API Constant final : public IntrusiveForwardNode<Constant, DerivedValue<Constant, DerivedValueTag::CONSTANT>> {

private:
    union {
        std::byte _small[sizeof(void *)] = {};
        void *_large;
    };
    uint64_t _hash = {};

    [[nodiscard]] bool _is_small() const noexcept;
    [[noreturn]] void _error_cannot_change_type() const noexcept;
    void _check_reinterpret_cast_type_size(size_t size) const noexcept;
    void _update_hash() noexcept;

public:
    explicit Constant(const Type *type, const void *data = nullptr) noexcept;
    ~Constant() noexcept override;

    [[noreturn]] void set_type(const Type *type) noexcept override {
        _error_cannot_change_type();
    }

    void set_data(const void *data) noexcept;
    [[nodiscard]] void *data() noexcept;
    [[nodiscard]] const void *data() const noexcept;

    void set_zero() noexcept;
    void set_one() noexcept;

    [[nodiscard]] auto hash() const noexcept { return _hash; }

    template<typename T>
    [[nodiscard]] T &as() noexcept {
        _check_reinterpret_cast_type_size(sizeof(T));
        return *static_cast<T *>(data());
    }

    template<typename T>
    [[nodiscard]] const T &as() const noexcept {
        return const_cast<Constant *>(this)->as<T>();
    }
};

using ConstantList = IntrusiveForwardList<Constant>;

}// namespace luisa::compute::xir
