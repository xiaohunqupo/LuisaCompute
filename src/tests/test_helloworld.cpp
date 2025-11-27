#define LUISA_ENABLE_DSL
#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
struct TestStruct {
    half2 a;
    bool b;
};
LUISA_STRUCT(TestStruct, a,b){};

int main(int argc, char *argv[]) {
    Context ctx(argv[0]);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    auto b = device.create_buffer<TestStruct>(1);
    auto s = device.compile<1>([&](Float4x4 f) {
        Var<TestStruct> t;
        b->write(0, t);
    });
    auto stream = device.create_stream();
    float4x4 r = make_float4x4(
        1, 2, 3, 4,
        5, 6, 7, 8,
        1, 2, 3, 4,
        5, 6, 7, 8);
    float4x4 result;
    stream << s(r).dispatch(1) << b.copy_to(&result) << synchronize();
    LUISA_INFO("{}", result);
}
