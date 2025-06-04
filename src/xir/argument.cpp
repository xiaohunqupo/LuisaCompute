#include <luisa/core/logging.h>
#include <luisa/xir/argument.h>

namespace luisa::compute::xir {

SentinelArgument::SentinelArgument(Function *parent_function) noexcept
    : Argument{parent_function, nullptr} {}

DerivedArgumentTag SentinelArgument::derived_argument_tag() const noexcept {
    LUISA_ERROR_WITH_LOCATION("Sentinel argument should not be used.");
}

}// namespace luisa::compute::xir
