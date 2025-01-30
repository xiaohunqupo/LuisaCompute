#include <luisa/xir/undefined.h>

namespace luisa::compute::xir {

Undefined *Undefined::create(const Type *type) noexcept {
    return Pool::current()->create<Undefined>(type);
}

}// namespace luisa::compute::xir
