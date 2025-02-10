#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

int main(int argc, char *argv[]) {

    auto context = luisa::compute::Context{luisa::current_executable_path()};
    auto device = context.create_device("fallback");

    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint>(1u);

    using namespace luisa::compute;
    auto shader = device.compile<1>([&](UInt n) noexcept {
        auto x = def<uint>();
        auto zero = def(0u);
        auto one = def(1u);
        $while (true) {
            $if (n == 0u) {
                x = zero;
            }
            $else {
                n -= 1u;
                x = one;
            };
            $if (n == 0u) {
                $break;
            };
        };
        buffer->write(0u, x);
    });

    uint result = 0u;
    stream << shader(10u).dispatch(1u)
           << buffer.copy_to(&result)
           << synchronize();

    LUISA_INFO("result = {}", result);
}
