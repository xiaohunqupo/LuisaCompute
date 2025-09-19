#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context ctx(argv[0]);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
}
