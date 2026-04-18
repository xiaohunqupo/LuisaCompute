#pragma once
#include <DXApi/LCCmdBuffer.h>
#include <Resource/RenderTexture.h>
#include <DXApi/LCDevice.h>
#include <Resource/SwapChain.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/DxPtr.h>
#include <dcomp.h>
namespace lc::dx {
class LCSwapChain : public Resource {
public:
    vstd::vector<SwapChain> render_targets;
    DxPtr<IDXGISwapChain1> swap_chain;
    uint64 frame_index = 0;
    uint64 frame_count = 0;
    DXGI_FORMAT format;
    ComPtr<IDCompositionDevice> dcomp_device;
    ComPtr<IDCompositionTarget> dcomp_target;
    ComPtr<IDCompositionVisual> dcomp_visual;
    bool vsync;
    Tag get_tag() const override { return Tag::SwapChain; }
    LCSwapChain(
        Device *device,
        CommandQueue *queue,
        GpuAllocator *resource_allocator,
        HWND window_handle,
        uint width,
        uint height,
        DXGI_FORMAT format,
        bool vsync,
        uint back_buffer_count,
        bool transparent = false);
    LCSwapChain(
        PixelStorage &storage,
        Device *device,
        IDXGISwapChain1 *swap_chain,
        bool vsync);
};
}// namespace lc::dx
