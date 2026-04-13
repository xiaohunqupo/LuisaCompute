#ifdef LUISA_TEST_DX_BACKEND

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/gui/window.h>
#include <luisa/backends/ext/dx_custom_cmd.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    log_level_verbose();
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: dx", argv[0]);
        return 1;
    }
    Context context{argv[0]};
    Device device = context.create_device(argv[1]);
    LUISA_INFO("Supersampling example: not yet implemented.");
    return 0;
}

#else

int main() {
    return 0;
}

#endif
