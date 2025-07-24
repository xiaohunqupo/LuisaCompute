#include "raster_shader.h"
#include "device.h"
#include "log.h"
namespace lc::vk {
RasterShader::RasterShader(
    Device *device,
    vstd::vector<Argument> &&captured,
    vstd::vector<SavedArgument> &&saved_arguments,
    vstd::span<hlsl::Property const> binds,
    vstd::span<std::byte const> cache_code,
    bool use_tex2d_bindless,
    bool use_tex3d_bindless,
    bool use_buffer_bindless)
    : Shader(device, ShaderTag::RasterShader, std::move(captured), std::move(saved_arguments), binds, use_tex2d_bindless, use_tex3d_bindless, use_buffer_bindless, {}) {
    VkPipelineCacheCreateInfo pso_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (!cache_code.empty()) {
        pso_ci.initialDataSize = cache_code.size();
        pso_ci.pInitialData = cache_code.data();
    }
    VK_CHECK_RESULT(vkCreatePipelineCache(device->logic_device(), &pso_ci, Device::alloc_callbacks(), &_pipe_cache));
}
RasterShader::~RasterShader() {
    for (auto &i : _pipelines) {
        vkDestroyPipeline(device()->logic_device(), i.second.pipeline, Device::alloc_callbacks());
        vkDestroyRenderPass(device()->logic_device(), i.second.render_pass, Device::alloc_callbacks());
    }
    vkDestroyPipelineCache(device()->logic_device(), _pipe_cache, Device::alloc_callbacks());
}
auto RasterShader::_make_pipeline_key(
    MeshFormat const &mesh_format,
    RasterState const &state,
    VkPipelineVertexInputStateCreateInfo &vertex_input_create_info) -> BinaryBlob {
    size_t key_size = 0;
    vertex_input_create_info = VkPipelineVertexInputStateCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    // compute size
    {
        // vertices
        vertex_input_create_info.vertexAttributeDescriptionCount = mesh_format.vertex_attribute_count();
        vertex_input_create_info.vertexBindingDescriptionCount = mesh_format.vertex_stream_count();
        key_size += vertex_input_create_info.vertexAttributeDescriptionCount * sizeof(VkVertexInputAttributeDescription) + vertex_input_create_info.vertexBindingDescriptionCount * sizeof(VkVertexInputBindingDescription);
        key_size += sizeof(RasterState);
    }
    if (key_size == 0) [[unlikely]]
        return {};
    BinaryBlob result(key_size);// make memory clear

    // push data
    {
        std::byte *ptr = result.data();
        luisa::fixed_vector<uint32_t, 16> offsets;
        offsets.reserve(mesh_format.vertex_stream_count());
        vertex_input_create_info.pVertexAttributeDescriptions = (const VkVertexInputAttributeDescription *)ptr;
        for (auto stream_idx : vstd::range(mesh_format.vertex_stream_count())) {
            uint32_t binding = 0;
            uint32_t offset = 0;
            for (auto &&attrs : mesh_format.attributes(stream_idx)) {
                VkVertexInputAttributeDescription desc{
                    .location = (uint32_t)stream_idx,
                    .binding = binding,
                    .format = Texture::to_vk_format(attrs.format),
                    .offset = offset};
                ++binding;
                offset += pixel_format_size(attrs.format, uint3(1));
                std::memcpy(ptr, &desc, sizeof(desc));
                ptr += sizeof(desc);
            }
            offsets.emplace_back(offset);
        }
        vertex_input_create_info.pVertexBindingDescriptions = (const VkVertexInputBindingDescription *)ptr;
        for (auto stream_idx : vstd::range(mesh_format.vertex_stream_count())) {
            VkVertexInputBindingDescription desc{
                .binding = (uint32_t)stream_idx,
                .stride = offsets[stream_idx],
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
            std::memcpy(ptr, &desc, sizeof(desc));
            ptr += sizeof(desc);
        }
        ptr += sizeof(RasterState);
        LUISA_ASSERT(ptr == result.data() + result.size());
    }
    return result;
}
void RasterShader::create_pipeline(
    vstd::span<uint const> spv_code,
    MeshFormat const &mesh_format,
    RasterState const &state) {
    VkPrimitiveTopology topo;
    VkCullModeFlags cull_mode;
    VkPipelineVertexInputStateCreateInfo vertex_input;
    auto binary_blob = _make_pipeline_key(
        mesh_format,
        state,
        vertex_input);
    auto iter = _pipelines.try_emplace(std::move(binary_blob));
    if (!iter.second) return;
    switch (state.cull_mode) {
        case CullMode::None:
            cull_mode = VK_CULL_MODE_NONE;
            break;
        case CullMode::Back:
            cull_mode = VK_CULL_MODE_BACK_BIT;
            break;
        case CullMode::Front:
            cull_mode = VK_CULL_MODE_FRONT_BIT;
            break;
    }
    switch (state.topology) {
        case TopologyType::Point:
            topo = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            break;
        case TopologyType::Line:
            topo = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            break;
        case TopologyType::Triangle:
            topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
    }
    VkPipelineInputAssemblyStateCreateInfo input_asm_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topo,
        .primitiveRestartEnable = VK_FALSE};

    VkPipelineRasterizationStateCreateInfo raster_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = state.depth_clip,
        .rasterizerDiscardEnable = false,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = cull_mode,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0,
        .depthBiasClamp = 1.f,
        .depthBiasSlopeFactor = 1.f,
        .lineWidth = 1.f};
    uint32_t sample_mask = -1;
    VkPipelineMultisampleStateCreateInfo ms_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0,
        .pSampleMask = &sample_mask,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE};
    auto get_blend_op = [](BlendOp op) {
        switch (op) {
            case BlendOp::Add:
                return VK_BLEND_OP_ADD;
            case BlendOp::Max:
                return VK_BLEND_OP_MAX;
            case BlendOp::Min:
                return VK_BLEND_OP_MIN;
            default:
                return VK_BLEND_OP_SUBTRACT;
        }
    };
    auto get_blend_factor = [](BlendWeight weight) {
        switch (weight) {
            case BlendWeight::Zero:
                return VK_BLEND_FACTOR_ZERO;
            case BlendWeight::One:
                return VK_BLEND_FACTOR_ONE;
            case BlendWeight::PrimColor:
                return VK_BLEND_FACTOR_SRC_COLOR;
            case BlendWeight::ImgColor:
                return VK_BLEND_FACTOR_DST_COLOR;
            case BlendWeight::PrimAlpha:
                return VK_BLEND_FACTOR_SRC_ALPHA;
            case BlendWeight::ImgAlpha:
                return VK_BLEND_FACTOR_DST_ALPHA;
            case BlendWeight::OneMinusPrimColor:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            case BlendWeight::OneMinusImgColor:
                return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
            case BlendWeight::OneMinusPrimAlpha:
                return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            default:
                return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        }
    };

    VkPipelineColorBlendAttachmentState blend_attachment{
        .blendEnable = state.blend_state.enable_blend,
    };
    blend_attachment.srcColorBlendFactor = get_blend_factor(state.blend_state.prim_op);
    blend_attachment.dstColorBlendFactor = get_blend_factor(state.blend_state.img_op);
    blend_attachment.colorBlendOp = get_blend_op(state.blend_state.op);
    blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blend_attachment.colorWriteMask = (VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
    VkPipelineColorBlendStateCreateInfo blend_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_CLEAR,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
        .blendConstants = {1.f, 1.f, 1.f, 1.f}};
    VkViewport view;
    VkRect2D scissors;
    VkPipelineViewportStateCreateInfo viewport_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &view,
        .scissorCount = 1,
        .pScissors = &scissors};
    VkPipelineTessellationStateCreateInfo tess_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .patchControlPoints = 1};
    // view.minDepth = 0;
    // view.maxDepth = 1;
    // view.x = viewport.start.x;
    // view.y = viewport.start.y;
    // view.width = viewport.size.x;
    // view.height = viewport.size.y;
    // scissors.offset.x = viewport.start.x;
    // scissors.offset.y = viewport.start.y;
    // scissors.extent.width = viewport.size.x;
    // scissors.extent.height = viewport.size.y;

    auto get_compare_op = [](Comparison const &comp) {
        switch (comp) {
            case Comparison::Never:
                return VK_COMPARE_OP_NEVER;
            case Comparison::Less:
                return VK_COMPARE_OP_LESS;
            case Comparison::Equal:
                return VK_COMPARE_OP_EQUAL;
            case Comparison::LessEqual:
                return VK_COMPARE_OP_LESS_OR_EQUAL;
            case Comparison::Greater:
                return VK_COMPARE_OP_GREATER;
            case Comparison::NotEqual:
                return VK_COMPARE_OP_NOT_EQUAL;
            case Comparison::GreaterEqual:
                return VK_COMPARE_OP_GREATER_OR_EQUAL;
            default:
                return VK_COMPARE_OP_ALWAYS;
        }
    };
    auto get_stencil_state = [](StencilOp s) {
        switch (s) {
            case StencilOp::Keep:
                return VK_STENCIL_OP_KEEP;
            case StencilOp::Replace:
                return VK_STENCIL_OP_REPLACE;
            default:
                return VK_STENCIL_OP_ZERO;
        }
    };
    VkPipelineDepthStencilStateCreateInfo depth_stencil_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = state.depth_state.enable_depth,
        .depthWriteEnable = state.depth_state.write,
        .depthCompareOp = get_compare_op(state.depth_state.comparison),
        .depthBoundsTestEnable = VK_TRUE,
        .stencilTestEnable = state.stencil_state.enable_stencil,
        // .front  = state.stencil_state.front_face_op
        // TODO
    };
    depth_stencil_info.front.failOp = get_stencil_state(state.stencil_state.front_face_op.stencil_fail_op);
    depth_stencil_info.front.passOp = get_stencil_state(state.stencil_state.front_face_op.pass_op);
    depth_stencil_info.front.depthFailOp = get_stencil_state(state.stencil_state.front_face_op.depth_fail_op);
    depth_stencil_info.front.compareOp = get_compare_op(state.stencil_state.front_face_op.comparison);
    depth_stencil_info.front.compareMask = ~0u;
    depth_stencil_info.front.writeMask = ~0u;
    depth_stencil_info.front.compareMask = ~0u;
    depth_stencil_info.front.reference = ~0u;

    depth_stencil_info.back.failOp = get_stencil_state(state.stencil_state.back_face_op.stencil_fail_op);
    depth_stencil_info.back.passOp = get_stencil_state(state.stencil_state.back_face_op.pass_op);
    depth_stencil_info.back.depthFailOp = get_stencil_state(state.stencil_state.back_face_op.depth_fail_op);
    depth_stencil_info.back.compareOp = get_compare_op(state.stencil_state.back_face_op.comparison);
    depth_stencil_info.back.compareMask = ~0u;
    depth_stencil_info.back.writeMask = ~0u;
    depth_stencil_info.back.compareMask = ~0u;
    depth_stencil_info.back.reference = ~0u;
    depth_stencil_info.minDepthBounds = 0.f;
    depth_stencil_info.maxDepthBounds = 1.f;
    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = vstd::array_count(dynamic_states),
        .pDynamicStates = dynamic_states};
    VkShaderModule shader_module;
    VkShaderModuleCreateInfo module_create_info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_code.size_bytes(),
        .pCode = spv_code.data()};
    VK_CHECK_RESULT(vkCreateShaderModule(device()->logic_device(), &module_create_info, Device::alloc_callbacks(), &shader_module));
    auto dispose_module = vstd::scope_exit([&] {
        vkDestroyShaderModule(device()->logic_device(), shader_module, Device::alloc_callbacks());
    });
    VkPipelineShaderStageCreateInfo vertex_stages[] = {
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shader_module,
            .pName = "main"}};
    auto &v = iter.first.value();
    // TODO render_pass
    VkGraphicsPipelineCreateInfo graphics_pipeline_create_info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = vstd::array_count(vertex_stages),
        .pStages = vertex_stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_asm_info,
        .pTessellationState = &tess_info,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_info,
        .pMultisampleState = &ms_info,
        .pDepthStencilState = &depth_stencil_info,
        .pColorBlendState = &blend_info,
        .pDynamicState = &dynamic_info,
        .layout = pipeline_layout(),
        .renderPass = v.render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = ~0};

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device()->logic_device(), _pipe_cache, 1, &graphics_pipeline_create_info, Device::alloc_callbacks(), &v.pipeline));
}
}// namespace lc::vk
