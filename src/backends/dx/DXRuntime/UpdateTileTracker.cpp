#include "UpdateTileTracker.h"

namespace lc::dx {
void UpdateTileTracker::record(
    ID3D12Heap *heap,
    ID3D12Resource *resource,
    D3D12_TILED_RESOURCE_COORDINATE const &ResourceRegionStartCoordinate,
    D3D12_TILE_REGION_SIZE const &ResourceRegionSize,
    D3D12_TILE_RANGE_FLAGS RangeFlag,
    UINT HeapRangeStartOffset,
    UINT RangeTileCount) {
    std::pair<ID3D12Heap *, ID3D12Resource *> key{
        heap,
        resource};
    auto iter = map.emplace(key);
    auto &v = iter.value();
    v.ResourceRegionStartCoordinates.emplace_back(ResourceRegionStartCoordinate);
    v.ResourceRegionSizes.emplace_back(ResourceRegionSize);
    v.RangeFlags.emplace_back(RangeFlag);
    v.HeapRangeStartOffsets.emplace_back(HeapRangeStartOffset);
    v.RangeTileCounts.emplace_back(RangeTileCount);
}
void UpdateTileTracker::update(
    ID3D12CommandQueue *queue,
    D3D12_TILE_MAPPING_FLAGS Flags) {
    for (auto &kv : map) {
        queue->UpdateTileMappings(
            kv.first.second,
            kv.second.ResourceRegionStartCoordinates.size(),
            kv.second.ResourceRegionStartCoordinates.data(),
            kv.second.ResourceRegionSizes.data(),
            kv.first.first, kv.second.RangeFlags.size(),
            kv.second.RangeFlags.data(),
            kv.second.HeapRangeStartOffsets.data(),
            kv.second.RangeTileCounts.data(),
            Flags);
    }
    for (auto &kv : disp_map) {
        queue->UpdateTileMappings(
            kv.first,
            kv.second.ResourceRegionStartCoordinates.size(),
            kv.second.ResourceRegionStartCoordinates.data(),
            kv.second.ResourceRegionSizes.data(),
            nullptr,
            1,
            vstd::get_rval_ptr(D3D12_TILE_RANGE_FLAG_NULL),
            nullptr,
            nullptr,
            Flags);
    }
    map.clear();
    disp_map.clear();
}
void UpdateTileTracker::deallocate(
    ID3D12Resource *resource,
    D3D12_TILED_RESOURCE_COORDINATE const &ResourceRegionStartCoordinate,
    D3D12_TILE_REGION_SIZE const &ResourceRegionSize) {
    auto iter = disp_map.emplace(resource);
    auto &v = iter.value();
    v.ResourceRegionStartCoordinates.emplace_back(ResourceRegionStartCoordinate);
    v.ResourceRegionSizes.emplace_back(ResourceRegionSize);
}
}// namespace lc::dx