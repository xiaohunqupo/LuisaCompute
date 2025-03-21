#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

int main(int argc, char *argv[]) {

    using namespace luisa::compute;

    Callable just_one = []() noexcept {
        return 1u;
    };



    Kernel1D kernel = [&](BufferUInt buffer, UInt n, AccelVar accel) noexcept {
        auto z = def(1u);
        $loop {
            UInt x;
            x = 2u;
            x = 2u;
            n -= just_one();
            $if (n == 1u) {
                $break;
            };
            z *= x;
        };
        device_log("你好, {} World!", z);
        buffer->write(0u, z);
        auto h = accel->traverse_any(make_ray(make_float3(), make_float3()), {}).trace();
    };

    auto ir = xir::ast_to_xir_translate(kernel.function()->function(), {});
    luisa::string dump;
    xir::XIRDebugPrinter{}.emit_module(dump, ir.get());
    LUISA_INFO("XIR Dump:\n{}", dump);
}
