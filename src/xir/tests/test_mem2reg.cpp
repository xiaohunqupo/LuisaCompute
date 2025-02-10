#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

int main(int argc, char *argv[]) {

    auto context = luisa::compute::Context{luisa::current_executable_path()};
    auto device = context.create_device("fallback");

    auto stream = device.create_stream();
    auto buffer = device.create_buffer<uint>(1u);

    using namespace luisa::compute;
    auto shader = device.compile<1>([&](UInt n) noexcept {
        auto b = compute::detail::FunctionBuilder::current();
        auto t = Type::of<uint>();
        auto x = b->local(t);
        auto zero = b->literal(t, 0u);
        auto one = b->literal(t, 1u);
        $if (n == 0u) {
        }
        $elif (n == 1u) {
            // b->assign(x, one);
        }
        $else {
            b->assign(x, zero);
        };
        buffer->write(0u, def<uint>(x));
    });

    uint result = 0u;
    stream << shader(10u).dispatch(1u)
           << buffer.copy_to(&result)
           << synchronize();

    LUISA_INFO("result = {}", result);
}
