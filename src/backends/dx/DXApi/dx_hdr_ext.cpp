#include <DXApi/dx_hdr_ext.hpp>
#include <DXApi/TypeCheck.h>
#include <LCAgilitySDK/d3d12.h>
#include <dxgi1_5.h>
#include <DXApi/LCSwapChain.h>
#include <Resource/TextureBase.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <Windows.h>
namespace lc::dx {
using namespace luisa::compute;
namespace dx_hdr_ext_detail {

DXHDRExt::DisplayCurve ensure_swap_chain_color_space(
    IDXGISwapChain4 *swap_chain,
    DXGI_COLOR_SPACE_TYPE &current_swap_chain_color_space,
    DXHDRExt::SwapChainBitDepth swap_chain_bit_depth,
    bool enable_st2084) noexcept {
    DXHDRExt::DisplayCurve result{DXHDRExt::DisplayCurve::None};
    DXGI_COLOR_SPACE_TYPE color_space = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    switch (swap_chain_bit_depth) {
        case DXHDRExt::SwapChainBitDepth::_8:
            result = DXHDRExt::DisplayCurve::sRGB;
            break;

        case DXHDRExt::SwapChainBitDepth::_10:
            color_space = enable_st2084 ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
            result = enable_st2084 ? DXHDRExt::DisplayCurve::ST2084 : DXHDRExt::DisplayCurve::sRGB;
            break;

        case DXHDRExt::SwapChainBitDepth::_16:
            color_space = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
            result = DXHDRExt::DisplayCurve::None;
            break;
    }

    if (current_swap_chain_color_space != color_space) {
        UINT color_space_support = 0;
        if (SUCCEEDED(swap_chain->CheckColorSpaceSupport(color_space, &color_space_support)) &&
            ((color_space_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) == DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
            ThrowIfFailed(swap_chain->SetColorSpace1(color_space));
            current_swap_chain_color_space = color_space;
        }
    }
    return result;
}
DXHDRExt::DisplayChromaticities set_hdr_meta_data(
    DXGI_COLOR_SPACE_TYPE &color_space,
    IDXGISwapChain4 *swapchain,
    DXGI_FORMAT format,
    bool hdr_support,
    float max_output_nits /*=1000.0f*/,
    float min_output_nits /*=0.001f*/,
    float max_cll /*=2000.0f*/,
    float max_fall /*=500.0f*/,
    const DXHDRExt::DisplayChromaticities *chroma) noexcept {
    if (!swapchain) {
        return {};
    }

    // Clean the hdr metadata if the display doesn't support HDR
    if (!hdr_support) {
        ThrowIfFailed(swapchain->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, 0, nullptr));
        return {};
    }

    static const DXHDRExt::DisplayChromaticities kDisplayChromaticityList[] =
        {
            {0.64000f, 0.33000f, 0.30000f, 0.60000f, 0.15000f, 0.06000f, 0.31270f, 0.32900f},// Display Gamut Rec709
            {0.70800f, 0.29200f, 0.17000f, 0.79700f, 0.13100f, 0.04600f, 0.31270f, 0.32900f},// Display Gamut Rec2020
        };

    // Select the chromaticity based on HDR format of the DWM.
    DXHDRExt::SwapChainBitDepth hit_depth = DXHDRExt::SwapChainBitDepth::_8;
    switch (format) {
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
        case DXGI_FORMAT_R10G10B10A2_UINT:
        case DXGI_FORMAT_R11G11B10_FLOAT:
            hit_depth = DXHDRExt::SwapChainBitDepth::_10;
            break;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_UNORM:
        case DXGI_FORMAT_R16G16B16A16_SNORM:
        case DXGI_FORMAT_R16G16B16A16_SINT:
        case DXGI_FORMAT_R16G16B16A16_UINT:
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            hit_depth = DXHDRExt::SwapChainBitDepth::_16;
            break;
        default:
            hit_depth = DXHDRExt::SwapChainBitDepth::_8;
            break;
    }

    ensure_swap_chain_color_space(swapchain, color_space, hit_depth, hdr_support);
    int selected_chroma = 0;
    if (format == DXGI_FORMAT_R16G16B16A16_FLOAT && color_space == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709) {
        selected_chroma = 0;
    } else if (hit_depth == DXHDRExt::SwapChainBitDepth::_10 && color_space == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
        selected_chroma = 1;
    } else {
        // Reset the metadata since this is not a supported HDR format.
        ThrowIfFailed(swapchain->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, 0, nullptr));
        return {};
    }

    // Set HDR meta data
    if (!chroma) {
        chroma = &kDisplayChromaticityList[selected_chroma];
    }
    DXGI_HDR_METADATA_HDR10 hdr10_meta_data = {};
    hdr10_meta_data.RedPrimary[0] = static_cast<UINT16>(chroma->RedX * 50000.0f);
    hdr10_meta_data.RedPrimary[1] = static_cast<UINT16>(chroma->RedY * 50000.0f);
    hdr10_meta_data.GreenPrimary[0] = static_cast<UINT16>(chroma->GreenX * 50000.0f);
    hdr10_meta_data.GreenPrimary[1] = static_cast<UINT16>(chroma->GreenY * 50000.0f);
    hdr10_meta_data.BluePrimary[0] = static_cast<UINT16>(chroma->BlueX * 50000.0f);
    hdr10_meta_data.BluePrimary[1] = static_cast<UINT16>(chroma->BlueY * 50000.0f);
    hdr10_meta_data.WhitePoint[0] = static_cast<UINT16>(chroma->WhiteX * 50000.0f);
    hdr10_meta_data.WhitePoint[1] = static_cast<UINT16>(chroma->WhiteY * 50000.0f);
    hdr10_meta_data.MaxMasteringLuminance = static_cast<UINT>(max_output_nits * 10000.0f);
    hdr10_meta_data.MinMasteringLuminance = static_cast<UINT>(min_output_nits * 10000.0f);
    hdr10_meta_data.MaxContentLightLevel = static_cast<UINT16>(max_cll);
    hdr10_meta_data.MaxFrameAverageLightLevel = static_cast<UINT16>(max_fall);
    ThrowIfFailed(swapchain->SetHDRMetaData(DXGI_HDR_METADATA_TYPE_HDR10, sizeof(DXGI_HDR_METADATA_HDR10), &hdr10_meta_data));
    return *chroma;
}
}// namespace dx_hdr_ext_detail
DXHDRExtImpl::DXHDRExtImpl(LCDevice *lc_device) : _lc_device(lc_device), _device_support_hdr(false) {
    UINT i = 0;
    ComPtr<IDXGIOutput> current_output;

    while (lc_device->native_device.adapter->EnumOutputs(i, &current_output) != DXGI_ERROR_NOT_FOUND) {
        // Having determined the output (display) upon which the app is primarily being
        // rendered, retrieve the HDR capabilities of that display by checking the color space.
        ComPtr<IDXGIOutput6> output6;
        ThrowIfFailed(current_output.As(&output6));
        DXGI_OUTPUT_DESC1 desc1;
        ThrowIfFailed(output6->GetDesc1(&desc1));
        _device_support_hdr |= (desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) || (desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709);
        i++;
    }
}


SwapchainCreationInfo DXHDRExtImpl::create_swapchain(
    const DXSwapchainOption &option,
    uint64_t stream_handle) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    if (queue->tag() != CmdQueueTag::MainCmd) [[unlikely]] {
        LUISA_ERROR("swapchain not allowed in Direct-Storage.");
    }
    SwapchainCreationInfo info{};
    auto res = new LCSwapChain(
        &(_lc_device->native_device),
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue,
        _lc_device->native_device.default_allocator.get(),
        reinterpret_cast<HWND>(option.window),
        option.size.x,
        option.size.y,
        static_cast<DXGI_FORMAT>(TextureBase::ToGFXFormat(pixel_storage_to_format<float>(option.storage))),
        option.wants_vsync,
        option.back_buffer_count);
    info.handle = resource_to_handle(res);
    info.native_handle = res->swap_chain.Get();
    info.storage = option.storage;
    return info;
}

auto DXHDRExtImpl::set_hdr_meta_data(
    uint64_t swapchain_handle,
    float max_output_nits,
    float min_output_nits,
    float max_cll,
    float max_fall,
    const DXHDRExt::DisplayChromaticities *custom_chroma) noexcept -> Meta {
    DXGI_COLOR_SPACE_TYPE color_space = DXGI_COLOR_SPACE_CUSTOM;
    auto swap_chain = reinterpret_cast<LCSwapChain *>(swapchain_handle);
    ComPtr<IDXGISwapChain4> swap_chain4;
    ThrowIfFailed(swap_chain->swap_chain->QueryInterface(IID_PPV_ARGS(&swap_chain4)));
    auto chroma = dx_hdr_ext_detail::set_hdr_meta_data(
        color_space,
        swap_chain4.Get(),
        swap_chain->format,
        true,
        max_output_nits,
        min_output_nits,
        max_cll,
        max_fall,
        custom_chroma);
    return {
        chroma};
}
void DXHDRExtImpl::set_color_space(
    uint64_t handle,
    ColorSpace const &color_space) const noexcept {
    auto swap_chain = reinterpret_cast<LCSwapChain *>(handle);
    ComPtr<IDXGISwapChain3> swap_chain3;
    ThrowIfFailed(swap_chain->swap_chain->QueryInterface(IID_PPV_ARGS(&swap_chain3)));
    swap_chain3->SetColorSpace1(static_cast<DXGI_COLOR_SPACE_TYPE>(color_space));
}
bool DXHDRExtImpl::device_support_hdr() const noexcept {
    return _device_support_hdr;
}
auto DXHDRExtImpl::get_display_data(uint64_t hwnd) const noexcept -> DisplayData {
    /////// Get window bound
    RECT window_rect = {};
    GetWindowRect(reinterpret_cast<HWND>(hwnd), &window_rect);

    UINT i = 0;
    ComPtr<IDXGIOutput> current_output;
    ComPtr<IDXGIOutput> best_output;
    float best_intersect_area = -1;
    auto compute_intersection_area = [](int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) -> float {
        return static_cast<float>(std::max(0, std::min(ax2, bx2) - std::max(ax1, bx1)) * std::max(0, std::min(ay2, by2) - std::max(ay1, by1)));
    };

    while (_lc_device->native_device.adapter->EnumOutputs(i, &current_output) != DXGI_ERROR_NOT_FOUND) {
        // Get the retangle bounds of the app window
        int ax1 = window_rect.left;
        int ay1 = window_rect.top;
        int ax2 = window_rect.right;
        int ay2 = window_rect.bottom;

        // Get the rectangle bounds of current output
        DXGI_OUTPUT_DESC desc;
        ThrowIfFailed(current_output->GetDesc(&desc));
        RECT r = desc.DesktopCoordinates;
        int bx1 = r.left;
        int by1 = r.top;
        int bx2 = r.right;
        int by2 = r.bottom;

        // Compute the intersection
        float intersect_area = compute_intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2);
        if (intersect_area > best_intersect_area) {
            best_output = current_output;
            best_intersect_area = intersect_area;
        }

        i++;
    }

    // Having determined the output (display) upon which the app is primarily being
    // rendered, retrieve the HDR capabilities of that display by checking the color space.
    ComPtr<IDXGIOutput6> output6;
    ThrowIfFailed(best_output.As(&output6));

    DXGI_OUTPUT_DESC1 desc1;
    ThrowIfFailed(output6->GetDesc1(&desc1));
    DisplayData display_data{
        .bits_per_color = desc1.BitsPerColor,
        .color_space = static_cast<ColorSpace>(desc1.ColorSpace),
        .red_primary = float2(desc1.RedPrimary[0], desc1.RedPrimary[1]),
        .green_primary = float2(desc1.GreenPrimary[0], desc1.GreenPrimary[1]),
        .blue_primary = float2(desc1.BluePrimary[0], desc1.BluePrimary[1]),
        .white_point = float2(desc1.WhitePoint[0], desc1.WhitePoint[1]),
        .min_luminance = desc1.MinLuminance,
        .max_luminance = desc1.MaxLuminance,
        .max_full_frame_luminance = desc1.MaxFullFrameLuminance};
    return display_data;
}
}// namespace lc::dx
