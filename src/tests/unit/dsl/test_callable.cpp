// Test for callable functions in the DSL
// This test demonstrates how to define and use reusable callable
// functions that can be composed and called from kernels.
//
// Features tested:
// - Callable function definition with auto parameters
// - Buffer read/write operations in callables
// - Callable composition (callables calling other callables)
// - Kernel using callables
// - Stream command list batching
// - Data transfer between host and device

#include <numeric>
#include <iostream>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/dsl/syntax.h>

using namespace luisa;
using namespace luisa::compute;

// Test structure with array member
struct Test {
    float a;
    float b;
    float array[16];
};

// Register the structure with the DSL
LUISA_STRUCT(Test, a, b, array) {};

int main(int argc, char *argv[]) {

    log_level_verbose();

    // Initialize context and device
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    static constexpr uint n = 1024u * 1024u;
    Device device = context.create_device(argv[1]);
    Buffer<float> buffer = device.create_buffer<float>(n);

    // Callable for loading values from buffer
    Callable load = [](BufferVar<float> buffer, Var<uint> index) noexcept {
        return buffer.read(index);
    };

    // Callable for storing values to buffer
    Callable store = [](BufferVar<float> buffer, Var<uint> index, Var<float> value) noexcept {
        buffer.write(index, value);
    };

    // Callable for simple arithmetic addition
    Callable add = [](Var<float> a, Var<float> b) noexcept {
        return a + b;
    };

    // Kernel that composes multiple callables
    Kernel1D kernel_def = [&](BufferVar<float> source, BufferVar<float> result, Var<float> x) noexcept {
        set_block_size(256u);
        UInt index = dispatch_id().x;
        // Chain callables: load -> add -> store
        auto xx = load(buffer, index);
        store(result, index, add(load(source, index), x) + xx);
    };
    auto kernel = device.compile(kernel_def);

    // Create stream and result buffer
    Stream stream = device.create_stream();
    Buffer<float> result_buffer = device.create_buffer<float>(n);

    // Prepare host data
    std::vector<float> data(n);
    std::vector<float> results(n);
    std::iota(data.begin(), data.end(), 1.0f);

    // Execute and time the kernel
    Clock clock;
    stream << buffer.copy_from(data.data());
    CommandList command_list = CommandList::create();
    // Dispatch kernel multiple times
    for (size_t i = 0; i < 10; i++) {
        command_list << kernel(buffer, result_buffer, 3).dispatch(n);
    }
    stream << command_list.commit()
           << result_buffer.copy_to(results.data());
    double t1 = clock.toc();
    stream << synchronize();
    double t2 = clock.toc();

    LUISA_INFO("Dispatched in {} ms. Finished in {} ms.", t1, t2);
    LUISA_INFO("Results: {}, {}, {}, {}, ..., {}, {}.",
               results[0], results[1], results[2], results[3],
               results[n - 2u], results[n - 1u]);

    // for (size_t i = 0u; i < n; i++) {
    //     LUISA_ASSERT(results[i] == data[i] + 3.0f, "Results mismatch.");
    // }
}
