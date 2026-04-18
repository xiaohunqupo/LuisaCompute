#include <DXApi/LCSwapChain.h>
#include <dxgi1_2.h>
#include <DXRuntime/Device.h>
#include <Resource/RenderTexture.h>
namespace lc::dx {
LCSwapChain::LCSwapChain(
    Device *device,
    CommandQueue *queue,
    GpuAllocator *resource_allocator,
    HWND window_handle,
    uint width,
    uint height,
    DXGI_FORMAT format,
    bool vsync,
    uint back_buffer_count,
    bool transparent)
    : Resource(device), vsync(vsync) {
    this->format = format;
    frame_count = back_buffer_count + 1;
    vstd::push_back_func(
        render_targets,
        frame_count,
        [device] { return device; });
    DXGI_SWAP_CHAIN_DESC1 swap_chain_desc = {};
    swap_chain_desc.BufferCount = frame_count;
    swap_chain_desc.Width = width;
    swap_chain_desc.Height = height;
    swap_chain_desc.Format = format;
    swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    if (!vsync)
        swap_chain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    swap_chain_desc.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    swap_chain_desc.SampleDesc.Count = 1;

    {
        IDXGISwapChain1 *local_swap;
        if (transparent) {
            swap_chain_desc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
            // For alpha-blended swapchains, use CreateSwapChainForComposition
            // Note: This requires DirectComposition to be set up by the caller
            ThrowIfFailed(device->dxgi_factory->CreateSwapChainForComposition(
                queue->queue(),
                &swap_chain_desc,
                nullptr,
                &local_swap));

            DCompositionCreateDevice(
                nullptr, IID_PPV_ARGS(&dcomp_device));
            dcomp_device->CreateTargetForHwnd(window_handle, true, &dcomp_target);
            dcomp_device->CreateVisual(&dcomp_visual);
            dcomp_visual->SetContent(local_swap);
            dcomp_target->SetRoot(dcomp_visual.Get());
            dcomp_device->Commit();
        } else {
            ThrowIfFailed(device->dxgi_factory->CreateSwapChainForHwnd(
                queue->queue(),
                window_handle,
                &swap_chain_desc,
                nullptr,
                nullptr,
                &local_swap));
        }
        swap_chain = DxPtr(local_swap, true);
    }
    for (uint32_t n = 0; n < frame_count; n++) {
        ThrowIfFailed(swap_chain->GetBuffer(n, IID_PPV_ARGS(&render_targets[n].rt)));
    }
    if (!vsync) {
        ComPtr<IDXGISwapChain3> swap_chain3;
        auto hr = swap_chain->QueryInterface(IID_PPV_ARGS(&swap_chain3));
        if (hr == S_OK) {
            swap_chain3->SetMaximumFrameLatency(back_buffer_count * 2);
        } else {
            LUISA_WARNING("Can not get IDXGISwapChain3, please check your Direct-X runtime.");
        }
    }
}
LCSwapChain::LCSwapChain(
    PixelStorage &storage,
    Device *device,
    IDXGISwapChain1 *swap_chain,
    bool vsync)
    : Resource(device),
      swap_chain(swap_chain, false),
      format(DXGI_FORMAT_UNKNOWN),
      vsync(vsync) {
    DXGI_SWAP_CHAIN_DESC1 swap_chain_desc;
    swap_chain->GetDesc1(&swap_chain_desc);
    vstd::push_back_func(
        render_targets,
        swap_chain_desc.BufferCount,
        [device] { return device; });
    frame_count = swap_chain_desc.BufferCount;
    for (uint32_t n = 0; n < swap_chain_desc.BufferCount; n++) {
        ThrowIfFailed(swap_chain->GetBuffer(n, IID_PPV_ARGS(&render_targets[n].rt)));
    }
    storage = pixel_format_to_storage(TextureBase::ToPixelFormat(static_cast<GFXFormat>(swap_chain_desc.Format)));
}
}// namespace lc::dx
