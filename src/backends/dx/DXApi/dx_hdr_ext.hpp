#pragma once
#include <luisa/backends/ext/dx_hdr_ext_interface.h>
#include <DXApi/LCDevice.h>
namespace lc::dx {

class DXHDRExtImpl : public luisa::compute::DXHDRExt {
    LCDevice *_lc_device;
    bool _device_support_hdr = false;
    bool device_support_hdr() const noexcept override;
public:
    DXHDRExtImpl(LCDevice *lc_device);
    ~DXHDRExtImpl() = default;
    SwapchainCreationInfo create_swapchain(
        const DXSwapchainOption &option,
        uint64_t stream_handle) noexcept override;
    Meta set_hdr_meta_data(
        uint64_t swapchain_handle,
        float max_output_nits,
        float min_output_nits,
        float max_cll,
        float max_fall,
        const DXHDRExt::DisplayChromaticities *custom_chroma) noexcept override;
    void set_color_space(
        uint64_t handle,
        ColorSpace const &color_space) const noexcept override;
    [[nodiscard]] DisplayData get_display_data(uint64_t hwnd) const noexcept override;
};
}// namespace lc::dx