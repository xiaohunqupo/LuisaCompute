#pragma once
#include <d3d12.h>
#include <luisa/vstl/common.h>
namespace lc::dx {
class UpdateTileTracker {
    struct UpdateTileCommand {
        vstd::fixed_vector<D3D12_TILED_RESOURCE_COORDINATE, 4>ResourceRegionStartCoordinates;
        vstd::fixed_vector<D3D12_TILE_REGION_SIZE, 4>ResourceRegionSizes;
        vstd::fixed_vector<D3D12_TILE_RANGE_FLAGS, 4>RangeFlags;
        vstd::fixed_vector<UINT, 4>HeapRangeStartOffsets;
        vstd::fixed_vector<UINT, 4>RangeTileCounts;
    };
    struct DisposeTileTracker {
        vstd::fixed_vector<D3D12_TILED_RESOURCE_COORDINATE, 4>ResourceRegionStartCoordinates;
        vstd::fixed_vector<D3D12_TILE_REGION_SIZE, 4>ResourceRegionSizes;
    };
    vstd::HashMap<std::pair<ID3D12Heap *, ID3D12Resource *>, UpdateTileCommand> map;
    vstd::HashMap<ID3D12Resource *, DisposeTileTracker> disp_map;
public:
    void record(
        ID3D12Heap *heap,
        ID3D12Resource *resource,
        D3D12_TILED_RESOURCE_COORDINATE const &ResourceRegionStartCoordinate,
        D3D12_TILE_REGION_SIZE const &ResourceRegionSize,
        D3D12_TILE_RANGE_FLAGS RangeFlag,
        UINT HeapRangeStartOffset,
        UINT RangeTileCount);
    void deallocate(
        ID3D12Resource *resource,
        D3D12_TILED_RESOURCE_COORDINATE const &ResourceRegionStartCoordinate,
        D3D12_TILE_REGION_SIZE const &ResourceRegionSize);
    void update(
        ID3D12CommandQueue *queue,
        D3D12_TILE_MAPPING_FLAGS Flags);
};
}// namespace lcdx