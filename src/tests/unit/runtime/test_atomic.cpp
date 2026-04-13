// Test for atomic operations on buffers and shared memory.
//
// This test verifies various atomic operation types:
// - fetch_add: Atomically add and return old value
// - fetch_sub: Atomically subtract and return old value
// - fetch_max: Atomically compute max and return old value
// - compare_exchange: Atomic compare-and-swap
//
// Atomic operations ensure thread-safe concurrent access to memory
// locations from multiple threads.

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

// Custom struct for testing atomic operations on complex types
struct Something {
    uint x;
    float3 v;
};

LUISA_STRUCT(Something, x, v) {};

int main(int argc, char *argv[]) {

    // Enable verbose logging
    log_level_verbose();

    // Initialize compute context
    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    // Create buffer for atomic counter test
    Buffer<uint> buffer = device.create_buffer<uint>(4u);
    
    // Create a buffer to hold the constant value (1u)
    Buffer<uint> constant_buffer = device.create_buffer<uint>(1);
    uint host_value = 1u;
    Stream stream = device.create_stream();
    stream << constant_buffer.copy_from(&host_value) << synchronize();
    
    // Kernel demonstrating atomic fetch_add and conditional write
    // This pattern can be used for counting unique events
    Kernel1D count_kernel = [&](BufferUInt counter_buffer) noexcept {
        // Atomically add 1 to buffer[3], returns old value
        Var x = buffer->atomic(3u).fetch_add(counter_buffer.read(0));
        
        // Only the first thread to increment writes 1 to buffer[0]
        // This demonstrates atomic counting with flag setting
        if_(x == 0u, [&] {
            buffer->write(0u, 1u);
        });
    };
    auto count = device.compile(count_kernel);

    // Initialize host buffer to zeros
    uint4 host_buffer = make_uint4(0u);

    // Performance test for atomic operations
    Clock clock;
    clock.tic();
    stream << buffer.copy_from(&host_buffer)
           << count(constant_buffer).dispatch(102400u)  // Launch many threads
           << buffer.copy_to(&host_buffer)
           << synchronize();
    double time = clock.toc();
    
    // Validate results:
    // - buffer[0] should be 1 (set by first thread)
    // - buffer[3] should be 102400 (total atomic increments)
    LUISA_INFO("Count: {} {}, Time: {} ms", host_buffer.x, host_buffer.w, time);
    LUISA_ASSERT(host_buffer.x == 1u && host_buffer.w == 102400u,
                 "Atomic operation failed.");

    // Test atomic operations on float buffers
    Buffer<float> atomic_float_buffer = device.create_buffer<float>(1u);
    
    // Kernel with atomic subtraction (via negative add)
    Kernel1D add_kernel = [&](BufferFloat buffer) noexcept {
        buffer.atomic(0u).fetch_sub(-1.f);  // fetch_sub with negative = addition
    };
    auto add_shader = device.compile(add_kernel);

    // Test atomic operations on vector components
    Kernel1D vector_atomic_kernel = [](BufferFloat3 buffer) noexcept {
        buffer.atomic(0u).x.fetch_add(1.f);  // Atomic add to x component
    };

    // Test atomic operations on matrix elements
    Kernel1D matrix_atomic_kernel = [](BufferFloat2x2 buffer) noexcept {
        buffer.atomic(0u)[1].x.fetch_add(1.f);  // Atomic add to [1][0] element
    };

    // Test atomic operations on nested array elements
    Kernel1D array_atomic_kernel = [](BufferVar<std::array<std::array<float4, 3u>, 5u>> buffer) noexcept {
        buffer.atomic(0u)[1][2][3].fetch_add(1.f);  // Atomic add to specific array element
    };

    // Test atomic operations on struct members
    Kernel1D struct_atomic_kernel = [](BufferVar<Something> buffer) noexcept {
        auto a = buffer.atomic(0u);
        a.v.x.fetch_max(1.f);  // Atomic max on struct member
        
        // Test shared memory atomics
        Shared<float> s{16};
        s.atomic(0).compare_exchange(0.f, 1.f);  // CAS on shared memory
    };

    // Validate float atomic addition
    float result = 0.f;
    stream << atomic_float_buffer.copy_from(&result)
           << add_shader(atomic_float_buffer).dispatch(1024u)
           << atomic_float_buffer.copy_to(&result)
           << synchronize();
    LUISA_INFO("Atomic float result: {}.", result);
    LUISA_ASSERT(result == 1024.f, "Atomic float operation failed.");
}
