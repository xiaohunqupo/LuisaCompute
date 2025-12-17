#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

struct Test1 {
    half a;
    uint16_t b;
};
LUISA_STRUCT(Test1, a, b) {};

int main(int argc, char *argv[]) {
    Context ctx(argv[0]);
    Device device = ctx.create_device(argv[1]);
    auto bf = device.create_buffer<Test1>(1);
    Test1 t;
    auto s = device.compile<1>([&](){
        Var<Test1> tt;
        tt.a = 1.5f;
        tt.b = 66;
        tt.b = tt.b + tt.b;
        bf->write(0, tt);
    });

    auto stream = device.create_stream();
    stream << s().dispatch(1) << bf.copy_to(&t) << synchronize();
    LUISA_INFO("{}, {}", (float)t.a, (int)t.b);
}
