#include <cmath>
#include <luisa/luisa-compute.h>
#include "luisa/core/basic_traits.h"
#include "luisa/dsl/resource.h"
#include "luisa/dsl/stmt.h"
#include "luisa/dsl/struct.h"
#include <cstddef>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/var.h>
#include <luisa/dsl/builtin.h>

namespace luisa::parallel_primitive {

template<typename Type4Byte>
luisa::compute::Var<Type4Byte> ShuffleDown(luisa::compute::Var<Type4Byte> &input,
                                           luisa::compute::UInt curr_lane_id,
                                           luisa::compute::UInt offset,
                                           luisa::compute::UInt last_lane = 32u) {
    luisa::compute::UInt src_lane = curr_lane_id + offset;
    luisa::compute::Var<Type4Byte> result = compute::warp_read_lane(input, src_lane);
    $if (src_lane > last_lane) {
        result = input;
    };
    return result;
};

inline luisa::compute::UInt get_lane_mask_ge(luisa::compute::UInt lane_id,
                                             luisa::compute::UInt wave_size) {
    luisa::compute::ULong mask64 = ~((1ull << lane_id) - 1ull);
    mask64 &= (1ull << wave_size) - 1ull;
    return static_cast<luisa::compute::UInt>(mask64);
}

namespace details {
using namespace luisa::compute;

template<typename Type4Byte, size_t LOGIC_WARP_SIZE = 32>
struct WarpReduceShfl {
    constexpr static bool IS_ARCH_WARP = (LOGIC_WARP_SIZE == 32);

    template<typename ReduceOp>
    Var<Type4Byte> Reduce(const Var<Type4Byte> &input, ReduceOp op, UInt valid_item = LOGIC_WARP_SIZE) {
        Var<Type4Byte> result = input;
        compute::UInt lane_id = compute::warp_lane_id();
        compute::UInt wave_size = compute::warp_lane_count();

        compute::UInt offset = 1u;
        $while (offset < wave_size) {
            Var<Type4Byte> temp = ShuffleDown(result, lane_id, offset, valid_item);
            $if (lane_id + offset < valid_item) {
                result = op(result, temp);
            };
            offset <<= 1;
        };
        return result;
    }

    template<bool HEAD_SEGMENT, typename FlagT, typename ReduceOp>
    Var<Type4Byte> SegmentReduce(const Var<Type4Byte> &input,
                                 const Var<FlagT> &flag,
                                 ReduceOp redecu_op,
                                 const UInt &valid_item = LOGIC_WARP_SIZE) {
        compute::UInt lane_id = compute::warp_lane_id();
        compute::UInt wave_size = compute::warp_lane_count();

        UInt warp_flags = compute::warp_active_bit_mask(flag == 1).x;
        if constexpr (HEAD_SEGMENT) {
            warp_flags >>= 1;
        };

        if constexpr (!IS_ARCH_WARP) {
            compute::UInt member_mask = warp_mask<LOGIC_WARP_SIZE>(lane_id);
            compute::UInt warp_id = lane_id / compute::UInt(LOGIC_WARP_SIZE);
            warp_flags = (warp_flags & member_mask) >> (warp_id * UInt(LOGIC_WARP_SIZE));
        };

        warp_flags &= get_lane_mask_ge(lane_id, wave_size);

        warp_flags |= 1u << (wave_size - 1u);

        UInt last_lane = compute::clz(compute::reverse(warp_flags));

        return Reduce(input, redecu_op, last_lane + 1);
    }
};
}// namespace details

template<typename ScanOp>
struct SwizzleScanOp {
    ScanOp scan_op;

public:
    explicit SwizzleScanOp(ScanOp scan_op)
        : scan_op(scan_op) {
    }

    template<typename Type>
    compute::Var<Type> operator()(const compute::Var<Type> &a, const compute::Var<Type> &b) const noexcept {
        compute::Var<Type> _a = a;
        compute::Var<Type> _b = b;
        return scan_op(_b, _a);
    }
};

template<typename T>
using SmemType = luisa::compute::Shared<T>;
template<typename T>
using SmemTypePtr = luisa::compute::Shared<T> *;

enum class ScanTileStatus : uint {
    SCAN_TILE_OBB,         // out-of-bounds
    SCAN_TILE_INVALID = 99,// not yet valid
    SCAN_TILE_PARTIAL,     // tile aggregate is available
    SCAN_TILE_INCLUSIVE,   // inclusive tile prefix is available
};

template<typename StatusWord, typename T>
struct TileDescriptor {
    StatusWord status;
    T value;

    TileDescriptor()
        : status(static_cast<StatusWord>(ScanTileStatus::SCAN_TILE_INVALID)), value(T(0)) {};

    TileDescriptor(const TileDescriptor &) = default;
    TileDescriptor &operator=(const TileDescriptor &) = default;
};

// device and host
template<typename T>
struct ScanTileState {
    static_assert(sizeof(T) == 8 || sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1,
                  "Unsupported type size for ScanTileState");

    using StatusWord =
        std::conditional_t<sizeof(T) == 8, ulong, std::conditional_t<sizeof(T) == 4, uint, std::conditional_t<sizeof(T) == 2, ushort, uchar>>>;

    using TileDescriptorT = TileDescriptor<StatusWord, T>;
    TileDescriptorT d_tile_descriptions;
};

template<typename T, bool SINGLE_WORD = std::is_integral_v<T>>
struct ScanTileStateViewer;
template<typename T>
struct ScanTileStateViewer<T, true> {

    using StatusWordT = typename ScanTileState<T>::StatusWord;
    using TileDescriptorT = typename ScanTileState<T>::TileDescriptorT;

    constexpr static size_t TILE_STATUS_PADDING = 32;

    static void InitializeWardStatus(compute::BufferVar<ScanTileState<T>> &tile_state,
                                     compute::UInt num_tile) noexcept {

        compute::UInt tile_idx = compute::dispatch_id().x;
        compute::Var<ScanTileState<T>> state;

        $if (tile_idx < num_tile) {
            state.d_tile_descriptions.status =
                compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_INVALID));
            state.d_tile_descriptions.value = T(0);
            tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_idx, state);
        };
        $if (compute::block_id().x == 0 & compute::thread_x() < compute::UInt(TILE_STATUS_PADDING)) {
            state.d_tile_descriptions.status =
                compute::def(StatusWordT(ScanTileStatus::SCAN_TILE_OBB));
            state.d_tile_descriptions.value = T(0);
            tile_state.write(compute::thread_x(), state);
        };
    };

    static void SetInclusive(compute::BufferVar<ScanTileState<T>> &tile_state,
                             compute::UInt tile_index,
                             const compute::Var<T> &tile_prefix) noexcept {
        compute::Var<ScanTileState<T>> state;
        state.d_tile_descriptions.status = StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE);
        state.d_tile_descriptions.value = tile_prefix;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };

    static void SetPartial(compute::BufferVar<ScanTileState<T>> &tile_state,
                           compute::UInt tile_index,
                           const compute::Var<T> &tile_partial) noexcept {
        compute::Var<ScanTileState<T>> state;
        state.d_tile_descriptions.status = StatusWordT(ScanTileStatus::SCAN_TILE_PARTIAL);
        state.d_tile_descriptions.value = tile_partial;
        tile_state.write(compute::UInt(TILE_STATUS_PADDING) + tile_index, state);
    };

    static void WaitForValid(compute::BufferVar<ScanTileState<T>> &tile_state,
                             compute::Int tile_index,
                             compute::Var<StatusWordT> &out_status,
                             compute::Var<T> &out_value) noexcept {
        compute::Var<ScanTileState<T>> curr_tile_state;
        curr_tile_state =
            tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        $while (compute::warp_active_any(curr_tile_state.d_tile_descriptions.status == StatusWordT(ScanTileStatus::SCAN_TILE_INVALID))) {
            curr_tile_state =
                tile_state.volatile_read(compute::Int(TILE_STATUS_PADDING) + tile_index);
        };

        out_status = curr_tile_state.d_tile_descriptions.status;
        out_value = curr_tile_state.d_tile_descriptions.value;

        $if (tile_index > 0) {

            compute::device_log("thid:{} Tile {} status: {}, value: {}", compute::dispatch_id().x, tile_index, out_status, out_value);
        };
    };
};

// Decoupled look-back(warp)
// only device
template<typename T, typename ScanOpT, typename ScanTileStateT>
class TilePrefixCallbackOp {
public:

    using StatusWordT = typename ScanTileStateT::StatusWord;

    compute::BufferVar<ScanTileStateT> &tile_status;
    ScanOpT scan_op;
    compute::Int tile_index;
    compute::Var<T> exclusive_prefix;
    compute::Var<T> inclusive_prefix;

    TilePrefixCallbackOp(compute::BufferVar<ScanTileStateT> &tile_state,
                         ScanOpT scan_op,
                         const compute::Int tile_index)
        : tile_status{tile_state}, scan_op{scan_op}, tile_index{tile_index} {};

    TilePrefixCallbackOp(compute::BufferVar<ScanTileStateT> &tile_state,
                         ScanOpT scan_op)
        : TilePrefixCallbackOp(tile_state, scan_op, compute::block_x()) {};

public:
    compute::Var<T> operator()(const compute::Var<T> &block_aggregate) {
        $if (compute::thread_x() == 0) {
            ScanTileStateViewer<T>::SetPartial(tile_status, tile_index, block_aggregate);
        };

        compute::Int predecessor_idx = tile_index - compute::thread_x() - 1;
        compute::Var<StatusWordT> predecessor_status;
        compute::Var<T> windows_aggregate;
        process_windows(predecessor_idx, predecessor_status, windows_aggregate);

        exclusive_prefix = windows_aggregate;

        $while (compute::warp_active_all(
                    predecessor_status != StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE))) {
            predecessor_idx -= compute::Int(32);
            process_windows(predecessor_idx, predecessor_status, windows_aggregate);
            exclusive_prefix = scan_op(windows_aggregate, exclusive_prefix);
        };

        $if (compute::thread_x() == 0) {
            inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
            ScanTileStateViewer<T>::SetInclusive(tile_status, tile_index, inclusive_prefix);
        };

        return exclusive_prefix;
    }

    compute::UInt GetTileIndex() const noexcept { return tile_index; }

private:
    void process_windows(compute::Int predecessor_idx,
                         compute::Var<StatusWordT> &predecessor_status,
                         compute::Var<T> &windows_aggregate) {
        compute::Var<T> value;
        ScanTileStateViewer<T>::WaitForValid(
            tile_status, predecessor_idx, predecessor_status, value);

        compute::Int tail_flag =
            predecessor_status == StatusWordT(ScanTileStatus::SCAN_TILE_INCLUSIVE);

        windows_aggregate = details::WarpReduceShfl<T>().template SegmentReduce<false>(
            value, tail_flag, scan_op);
    }
};

}// namespace luisa::parallel_primitive

#define LUISA_TILEDESCRIPTOR_TEMPLATE() \
    template<typename StatusWord, typename U>
#define LUISA_TILEDESCRIPTOR_NAME() \
    luisa::parallel_primitive::TileDescriptor<StatusWord, U>
LUISA_TEMPLATE_STRUCT(LUISA_TILEDESCRIPTOR_TEMPLATE, LUISA_TILEDESCRIPTOR_NAME, status, value){};

#define LUISA_T_TEMPLATE() template<typename U>

#define LUISA_SCANTILESTATE_TRUE_NAME() \
    luisa::parallel_primitive::ScanTileState<U>
LUISA_TEMPLATE_STRUCT(LUISA_T_TEMPLATE, LUISA_SCANTILESTATE_TRUE_NAME, d_tile_descriptions){};

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;

int main(int argc, char **argv) {
    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    CommandList cmdlist;
    Stream stream = device.create_stream();

    constexpr size_t WARP_SIZE = 32;
    constexpr size_t array_size = 256;
    constexpr size_t BLOCK_SIZE = 256;
    constexpr size_t NUM_TILES = 100;
    const size_t num_blocks = ceil(float(NUM_TILES) / BLOCK_SIZE);
    auto scan_tile_buffer = device.create_buffer<ScanTileState<int>>(WARP_SIZE + NUM_TILES);

    auto exclusive_buffer = device.create_buffer<int>(NUM_TILES);
    auto inclusive_buffer = device.create_buffer<int>(NUM_TILES);

    // init status to invalid and obb
    Kernel1D init_kernel = [&](BufferVar<ScanTileState<int>> tile_state) noexcept {
        luisa::parallel_primitive::ScanTileStateViewer<int, true>::InitializeWardStatus(tile_state, NUM_TILES);
    };
    auto init_shader = device.compile(init_kernel);
    cmdlist << init_shader(scan_tile_buffer.view()).dispatch(num_blocks);
    stream << cmdlist.commit() << synchronize();

    auto scan_op = [](const Var<int> &a, const Var<int> &b) noexcept { return a + b; };

    Kernel1D decoupled_look_back_kernel = [&](BufferVar<ScanTileState<int>> tile_state,
                                              BufferVar<int> exclusive_output,
                                              BufferVar<int> inclusive_output) noexcept {
        luisa::compute::set_block_size(BLOCK_SIZE);
        luisa::compute::set_warp_size(WARP_SIZE);
        compute::UInt tid = compute::thread_x();
        using tile_prefix_op = TilePrefixCallbackOp<int, decltype(scan_op), ScanTileState<int>>;

        tile_prefix_op prefix(tile_state, scan_op);
        const auto tile_idx = prefix.GetTileIndex();
        compute::Int block_aggregate = block_id().x;
        $if (tile_idx == 0) {
            $if (tid == 0) {
                luisa::parallel_primitive::ScanTileStateViewer<int, true>::SetInclusive(tile_state, tile_idx, block_aggregate);
                exclusive_output.write(tile_idx, 0);
                inclusive_output.write(tile_idx, 0);
            };
        }
        $else {
            const auto warp_id = tid / luisa::compute::UInt(WARP_SIZE);
            $if (warp_id == 0) {
                Var<int> exclusive_prefix = prefix(block_aggregate);
                $if (tid == 0) {
                    Var<int> inclusive_prefix =
                        scan_op(exclusive_prefix, block_aggregate);
                    exclusive_output.write(tile_idx, exclusive_prefix);
                    inclusive_output.write(tile_idx, inclusive_prefix);
                };
            };
        };
    };

    auto decoupled_look_back_kernel_shader = device.compile(decoupled_look_back_kernel);

    cmdlist
        << decoupled_look_back_kernel_shader(scan_tile_buffer.view(), exclusive_buffer.view(), inclusive_buffer.view()).dispatch(NUM_TILES * BLOCK_SIZE);
    stream << cmdlist.commit() << synchronize();

    luisa::vector<int> exclusive_result(NUM_TILES);
    luisa::vector<int> inclusive_result(NUM_TILES);
    stream << exclusive_buffer.copy_to(exclusive_result.data())
           << inclusive_buffer.copy_to(inclusive_result.data()) << synchronize();

    for (size_t i = 0; i < NUM_TILES; i++) {
        LUISA_INFO("Tile {}: exclusive_result = {}, inclusive_result = {}",
                   i,
                   exclusive_result[i],
                   inclusive_result[i]);
    }
}