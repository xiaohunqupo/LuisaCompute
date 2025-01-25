#include <luisa/xir/instructions/raster_discard.h>

namespace luisa::compute::xir {

RasterDiscardInst *RasterDiscardInst::clone(InstructionCloneValueResolver &resolver) const noexcept {
    return Pool::current()->create<RasterDiscardInst>();
}

}// namespace luisa::compute::xir
