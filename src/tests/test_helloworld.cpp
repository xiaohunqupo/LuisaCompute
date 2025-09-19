#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
struct Test {
    float4x4 proj;
    bool hdr;
    bool hdr_10;
    float lum;
    double random_noise_intensity;
    uint frame_index;
};
LUISA_STRUCT(Test, proj, hdr, hdr_10, lum, random_noise_intensity, frame_index) {};

int main(int argc, char *argv[]) {
    Context ctx(argv[0]);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    // auto shader = device.compile<1>(
    //     [](BufferVar<double4x4> b,
    //        Double4x4 view,
    //        Float2 a,
    //        Var<double> c,
    //        Bool e,
    //        Bool e1,
    //        Double4x4 proj) {
    //         auto p = view->operator*(proj);
    //         b.write(0, p);
    //     });
    // auto stream = device.create_stream();
    // auto buffer = device.create_buffer<double4x4>(1);
    // double4x4 r{};
    // stream << shader(buffer, make_double4x4(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4), float2(), 0.0, true, false, make_double4x4(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4))
    //               .dispatch(1)
    //        << buffer.copy_to(&r) << synchronize();
    // LUISA_INFO("{}", r);
}
