// Hello World example demonstrating basic buffer operations
// and half-precision floating point support.

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

// Test structure with half-precision float
struct Test1 {
    half a;
    uint16_t b;
};
LUISA_STRUCT(Test1, a, b) {};

int main(int argc, char *argv[]) {
    // Create context and device
    Context ctx(argv[0]);
    Device device = ctx.create_device(argv[1]);
    
    // Create a buffer to store Test1 structure
    auto bf = device.create_buffer<Test1>(1);
    Test1 t;
    
    // Compile a kernel that writes to the buffer
    auto s = device.compile<1>([&](){
        Var<Test1> tt;
        tt.a = 1.5f;
        tt.b = 66;
        tt.b = tt.b + tt.b;
        bf->write(0, tt);
    });

    // Create stream and dispatch kernel
    auto stream = device.create_stream();
    stream << s().dispatch(1) << bf.copy_to(luisa::span{&t, 1}) << synchronize();
    
    // Output results
    LUISA_INFO("{}, {}", (float)t.a, (int)t.b);
}
