#include <luisa/core/logging.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/device.h>
#include <luisa/dsl/sugar.h>
#include <random>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context ctx(argv[0]);
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = ctx.create_device(argv[1]);
    auto stream = device.create_stream();
    auto get_matrix = [&](BufferVar<float> const &buffer, UInt2 const &size, UInt2 const &idx) {
        return buffer.read(size.x * idx.y + idx.x);
    };
    auto set_matrix = [&](BufferVar<float> const &buffer, UInt2 const &size, UInt2 const &idx, Float const &value) {
        return buffer.write(size.x * idx.y + idx.x, value);
    };
    constexpr uint k_warp_size = 32;
    auto mat_mul_kernel = [&](BufferVar<float> lhs_buffer,
                              BufferVar<float> rhs_buffer,
                              BufferVar<float> result_buffer,
                              UInt lhs_row_size) {
        set_block_size(128, 1, 1);
        // Only work at device support Shader-Model 6.6 or any CUDA device
        // Implemented defined warp size for modern GPU
        set_warp_size(k_warp_size);
        UInt2 lhs_matrix_size = make_uint2(lhs_row_size, dispatch_size().y);
        UInt2 rhs_matrix_size = make_uint2(dispatch_size().x / k_warp_size, lhs_row_size);
        UInt lhs_y = dispatch_id().x / k_warp_size;
        UInt rhs_x = dispatch_id().y;
        UInt warp_local_id = warp_lane_id();
        UInt lhs_row_batch_count = (lhs_matrix_size.x + k_warp_size - 1) / k_warp_size;
        Float curr_lane_value;
        curr_lane_value = 0.f;

        Float local_v;
        for (auto lhs_row_batch : dynamic_range(lhs_row_batch_count)) {
            UInt lhs_x = lhs_row_batch * k_warp_size + warp_local_id;
            $if (lhs_x < lhs_matrix_size.x) {
                local_v = get_matrix(lhs_buffer, lhs_matrix_size, make_uint2(lhs_x, lhs_y));
                local_v *= get_matrix(rhs_buffer, rhs_matrix_size, make_uint2(rhs_x, lhs_x));
            }
            $else {
                local_v = 0.f;
            };
            curr_lane_value += warp_active_sum(local_v);
        }
        $if (warp_local_id == 0) {
            set_matrix(result_buffer, make_uint2(rhs_matrix_size.x, lhs_matrix_size.y), make_uint2(rhs_x, lhs_y), curr_lane_value);
        };
    };
    auto mat_mul_shader = device.compile<2>(std::move(mat_mul_kernel));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.5, 1.5);
    constexpr uint k_matrix_size = 256;
    luisa::vector<float> lhs_matrix;
    luisa::vector<float> rhs_matrix;
    luisa::vector<float> result_matrix;
    lhs_matrix.resize(k_matrix_size * k_matrix_size);
    rhs_matrix.resize(k_matrix_size * k_matrix_size);
    result_matrix.resize(k_matrix_size * k_matrix_size);
    auto idx = [](auto x, auto y) {
        return y * k_matrix_size + x;
    };
    for (int x = 0; x < k_matrix_size; ++x)
        for (int y = 0; y < k_matrix_size; ++y) {
            lhs_matrix[idx(x, y)] = dist(gen);
            rhs_matrix[idx(x, y)] = dist(gen);
        }
    auto lhs_buffer = device.create_buffer<float>(lhs_matrix.size());
    auto rhs_buffer = device.create_buffer<float>(rhs_matrix.size());
    auto result_buffer = device.create_buffer<float>(result_matrix.size());
    stream
        << lhs_buffer.copy_from(lhs_matrix.data())
        << rhs_buffer.copy_from(rhs_matrix.data())
        << mat_mul_shader(lhs_buffer, rhs_buffer, result_buffer, k_matrix_size).dispatch(k_matrix_size * k_warp_size, k_matrix_size) << result_buffer.copy_to(result_matrix.data()) << synchronize();
    // Host calculation validation
    for (int x = 0; x < k_matrix_size; ++x) {
        for (int y = 0; y < k_matrix_size; ++y) {
            float result = 0.f;
            for (int row = 0; row < k_matrix_size; ++row) {
                result += lhs_matrix[idx(row, y)] * rhs_matrix[idx(x, row)];
            }
            if (abs(result - result_matrix[idx(x, y)]) > 1e-2f) [[unlikely]] {
                LUISA_ERROR("Bad result {} {} at index {},{}.", result, result_matrix[idx(x, y)], x, y);
            }
        }
    }
}
