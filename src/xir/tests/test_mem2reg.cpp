#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

int main(int argc, char *argv[]) {

    auto context = luisa::compute::Context{luisa::current_executable_path()};
    auto device = context.create_device("fallback");

    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint>(1u);

    using namespace luisa::compute;
    auto shader = device.compile<1>([&](UInt n) noexcept {
        auto z = def(make_uint2(1u));
        $for (i, n) {
            UInt x;
            x = 2u;
            $if (i == 0u) {
                x = 2u;
                $continue;
            };
            n -= 1u;
            $if (n == 1u) {
                x = dispatch_id().x;
                $break;
            };
            z.x *= x + i;
        };
        buffer->write(0u, z.x);
    });

    auto result = 0u;
    stream << shader(10u).dispatch(1u)
           << buffer.copy_to(&result)
           << synchronize();

    LUISA_INFO("result = {}", result);
}
