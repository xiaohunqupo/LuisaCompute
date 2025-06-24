#include "shader.h"
#include "log.h"
#include "device.h"
namespace lc::vk {
SavedArgument::SavedArgument(Type const *type) {
    if (luisa::to_underlying(type->tag()) < luisa::to_underlying(Type::Tag::BUFFER)) {
        structSize = type->size();
    }
}
Shader::Shader(
    Device *device,
    ShaderTag tag,
    vstd::vector<Argument> &&captured,
    vstd::vector<SavedArgument> &&saved_arguments,
    vstd::span<hlsl::Property const> binds,
    bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool use_buffer_bindless)
    : Resource{device}, _captured{std::move(captured)}, _saved_arguments(std::move(saved_arguments)),
      _use_tex2d_bindless(use_tex2d_bindless), _use_tex3d_bindless(use_tex3d_bindless), _use_buffer_bindless(use_buffer_bindless) {
    VkShaderStageFlagBits stage_bits = [&]() -> VkShaderStageFlagBits {
        switch (tag) {
            case ShaderTag::ComputeShader:
                return VK_SHADER_STAGE_COMPUTE_BIT;
            case ShaderTag::RasterShader:
                return static_cast<VkShaderStageFlagBits>(
                    VK_SHADER_STAGE_VERTEX_BIT |
                    VK_SHADER_STAGE_FRAGMENT_BIT);
            default:
                return VK_SHADER_STAGE_ALL;
        }
    }();
    vstd::vector<vstd::vector<VkDescriptorSetLayoutBinding>> bindings;
    for (auto &&i : binds) {
        bindings.resize(std::max<size_t>(bindings.size(), i.space_index + 1));
        auto &vec = bindings[i.space_index];
        vec.resize(std::max<size_t>(vec.size(), i.register_index + 1));
        auto &v = vec[i.register_index];
        v.pImmutableSamplers = nullptr;
        v.binding = i.register_index;
        switch (i.type) {
            case hlsl::ShaderVariableType::ConstantBuffer:
            case hlsl::ShaderVariableType::ConstantValue:
            case hlsl::ShaderVariableType::CBVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case hlsl::ShaderVariableType::SRVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case hlsl::ShaderVariableType::UAVTextureHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case hlsl::ShaderVariableType::StructuredBuffer:
            case hlsl::ShaderVariableType::RWStructuredBuffer:
            case hlsl::ShaderVariableType::UAVBufferHeap:
            case hlsl::ShaderVariableType::SRVBufferHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case hlsl::ShaderVariableType::SPIRVAccel:
                v.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                break;
            case hlsl::ShaderVariableType::SamplerHeap:
                v.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                v.pImmutableSamplers = device->samplers().data();
                break;
            default:
                assert(false);
                break;
        }
        v.descriptorCount = i.array_size == ~0u ? 65536u : i.array_size;
        v.stageFlags = stage_bits;
    }
    vstd::push_back_all(_binds, binds);
    _desc_set_layout.reserve(bindings.size());
    for (auto &&i : bindings) {
        VkDescriptorSetLayoutCreateInfo descriptorLayout{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = static_cast<uint>(i.size()),
            .pBindings = i.data()};
        auto &r = _desc_set_layout.emplace_back();
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logic_device(), &descriptorLayout, Device::alloc_callbacks(), &r));
    }
    VkPushConstantRange push_const_range{
        VkShaderStageFlags(stage_bits),
        0,
        16};
    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint>(_desc_set_layout.size()),
        .pSetLayouts = _desc_set_layout.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_const_range};
    VK_CHECK_RESULT(
        vkCreatePipelineLayout(
            device->logic_device(),
            &pPipelineLayoutCreateInfo,
            Device::alloc_callbacks(),
            &_pipeline_layout));
}
Shader::~Shader() {
    for (auto &&i : _desc_set_layout) {
        vkDestroyDescriptorSetLayout(device()->logic_device(), i, Device::alloc_callbacks());
    }
    vkDestroyPipelineLayout(device()->logic_device(), _pipeline_layout, Device::alloc_callbacks());
}
}// namespace lc::vk
