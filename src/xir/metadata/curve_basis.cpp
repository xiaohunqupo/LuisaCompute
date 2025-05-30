#include <luisa/xir/metadata/curve_basis.h>

namespace luisa::compute::xir {

CurveBasisMD::CurveBasisMD(CurveBasisSet set) noexcept
    : _curve_basis_set{set} {}

ManagedPtr<Metadata> CurveBasisMD::clone() const noexcept {
    return luisa::make_managed<CurveBasisMD>(this->curve_basis_set());
}

}// namespace luisa::compute::xir
