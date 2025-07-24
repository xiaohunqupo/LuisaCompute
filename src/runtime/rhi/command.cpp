//
// Created by mike on 7/2/25.
//

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/command.h>

namespace luisa::compute {

void BindlessArrayUpdateCommand::report_unmatched_bindless_slot_type(luisa::string_view expected) noexcept {
    LUISA_ERROR_WITH_LOCATION("Unmatched bindless slot type (expected {}). ", expected);
}

}// namespace luisa::compute
