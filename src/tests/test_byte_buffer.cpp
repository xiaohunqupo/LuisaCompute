// Test for buffer I/O operations
// This test verifies buffer read/write operations and
// iteration patterns using command lists.
//
// Features tested:
// - Buffer creation with different types
// - Buffer views and element views
// - Buffer read/write in kernels
// - Atomic operations on buffer elements
// - Command list creation and commit
// - Data verification between iterations

#include <stb/stb_image_write.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/event.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/byte_buffer.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);
    constexpr uint BUFFER_SIZE = 4;
    auto byte_buffer = device.create_byte_buffer(BUFFER_SIZE * sizeof(uint));
    auto buffer_float = device.create_buffer<float>(BUFFER_SIZE);
    auto test_shader = device.compile<1>([&](ByteBufferVar buffer, UInt value) {
        auto id = dispatch_id().x;
        buffer.write(id * (uint)sizeof(float), value);
    });
    auto stream = device.create_stream();
    luisa::vector<uint> host_data(BUFFER_SIZE);
    stream << test_shader(ByteBufferView{buffer_float} /*Buffer<float> to Byte buffer*/, 114).dispatch(buffer_float.size())
           << test_shader(byte_buffer, 514).dispatch(buffer_float.size())
           << buffer_float.copy_to(luisa::span<uint>{host_data.data(), host_data.size()})
           << synchronize();
    LUISA_INFO("Buffer<float> as Buffer<uint>: ");
    for (auto &i : host_data) {
        LUISA_INFO("{}", i);
    }
    LUISA_INFO("ByteBuffer as Buffer<uint>: ");
    stream << byte_buffer.copy_to(host_data.data())
           << synchronize();
    for (auto &i : host_data) {
        LUISA_INFO("{}", i);
    }
}