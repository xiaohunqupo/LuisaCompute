// Device debugger test demonstrating programmable breakpoints
// and custom trap functions for GPU debugging.

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// Test structure with vector types
struct MyStruct {
    float2 a;
    uint2 b;
};

LUISA_STRUCT(MyStruct, a, b) {};

// Custom trap function with parameters for debugging
void my_trap(auto p, auto s, auto v, auto coord) {
    // You may do something with printing here
    LUISA_INFO("my_trap: p = {}, s = {}, v = {}, coord = {}", p, s, v, coord);
    // Please place a break point here with your IDE
}

// Custom trap function without parameters
void foo() {
    // Please place a break point here with your IDE
}

int main(int argc, char *argv[]) {

    log_level_verbose();

    // Create context and device (using fallback for debugging)
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    // Define a kernel with debug breakpoints
    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x == 1) {
            // Programmable break point with __debugbreak-like intrinsics
            $debug_break(coord);
        };
        $if (coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            Var<MyStruct> s;
            s.a = v;
            s.b = coord;
            // Break point with custom trap function (dispatch_id is always captured for convenience)
            $debug_break_on(s, v, coord, my_trap(dispatch_id, s, v, coord));
            $outline_with_name("my_logger") {
                device_log("s = {} at {}", s, dispatch_id());
            };
            // Custom trap function without parameters (useful for use with interactive debuggers)
            $debug_break_on(foo());
        };
    };

    // Compile and dispatch the kernel
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();
    stream << shader().dispatch(128u, 128u)
           << synchronize();
}
