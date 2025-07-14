#pragma once
#include "resource.h"
#include <volk.h>
#include "../common/hlsl/shader_property.h"
#include <luisa/runtime/rhi/argument.h>
#include "buffer.h"
#include "texture.h"
namespace lc::vk {
using namespace luisa::compute;
struct SavedArgument {
    Type::Tag tag;
    Usage varUsage;
    uint structSize;
    SavedArgument() {}
    SavedArgument(Function kernel, Variable const &var) : SavedArgument(var.type()) {
        varUsage = kernel.variable_usage(var.uid());
    }
    SavedArgument(Usage usage, Variable const &var) : SavedArgument(var.type()) {
        varUsage = usage;
    }
    SavedArgument(Type const *type);
};
class Shader : public Resource {
public:
    enum class ShaderTag : uint {
        ComputeShader,
        RasterShader
    };

protected:
    vstd::vector<VkDescriptorSetLayout> _desc_set_layout;
    VkPipelineLayout _pipeline_layout;
    vstd::vector<hlsl::Property> _binds;
    vstd::vector<Argument> _captured;
    vstd::vector<SavedArgument> _saved_arguments;
    vstd::vector<std::pair<luisa::string, Type const *>> _printers;
    bool _use_tex2d_bindless;
    bool _use_tex3d_bindless;
    bool _use_buffer_bindless;
public:
    auto pipeline_layout() const { return _pipeline_layout; }
    virtual bool serialize_pso(vstd::vector<std::byte> &result) const { return false; }
    auto binds() const { return vstd::span<const hlsl::Property>{_binds}; }
    auto captured() const { return vstd::span<const Argument>{_captured}; }
    auto desc_set_layout() const { return vstd::span{_desc_set_layout}; }
    auto saved_arguments() const { return vstd::span{_saved_arguments}; }
    bool use_tex2d_bindless() const { return _use_tex2d_bindless; };
    bool use_tex3d_bindless() const { return _use_tex3d_bindless; };
    bool use_buffer_bindless() const { return _use_buffer_bindless; };
    auto printers() const { return luisa::span{_printers}; }
    Shader(
        Device *device,
        ShaderTag tag,
        vstd::vector<Argument> &&captured,
        vstd::vector<SavedArgument> &&saved_arguments,
        vstd::span<hlsl::Property const> binds,
        bool use_tex2d_bindless,
        bool use_tex3d_bindless,
        bool use_buffer_bindless,
        vstd::vector<std::pair<luisa::string, Type const *>> &&printers);
    virtual ~Shader();
    vstd::span<VkDescriptorSet> allocate_desc_set(VkDescriptorPool pool, vstd::vector<VkDescriptorSet> &descs) const;
    void update_desc_set(
        VkDescriptorSet set,
        vstd::vector<VkWriteDescriptorSet> &write_buffer,
        vstd::vector<VkImageView> &img_view_buffer,
        vstd::span<vstd::variant<BufferView, TexView>> texs);
};
}// namespace lc::vk
