#include "resource_barrier.h"
namespace lc::vk {
namespace detail {
static constexpr auto raster_stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
static constexpr VkPipelineStageFlagBits2 BarrierSyncMap[] = {
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // ComputeRead,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // ComputeAccelRead,
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,                                                    // ComputeUAV,
    VK_PIPELINE_STAGE_2_COPY_BIT,                                                              // CopySource,
    VK_PIPELINE_STAGE_2_COPY_BIT,                                                              // CopyDest,
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,                                  // BuildAccel,
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,                                   // CopyAccelSrc
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR,                                   // CopyAccelDst
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,//DepthRead
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,//DepthWrite
    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,                                                     //IndirectArgs
    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,                                            //VertexRead,
    VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,                                                       //  IndexRead,
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,                                           //  RenderTarget
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,                                  // AccelInstanceBuffer
    raster_stage,                                                                              // RasterRead
    raster_stage,                                                                              //RasterAccelRead
    raster_stage                                                                               //RasterUAV
};
static constexpr VkAccessFlagBits2 BarrierAccessMap[] = {
    VK_ACCESS_2_SHADER_READ_BIT,                     // ComputeRead,
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, // ComputeAccelRead,
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,            // ComputeUAV,
    VK_ACCESS_2_TRANSFER_READ_BIT,                   // CopySource,
    VK_ACCESS_2_TRANSFER_WRITE_BIT,                  // CopyDest,
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,// BuildAccel,
    VK_ACCESS_2_TRANSFER_READ_BIT,                   // CopyAccelSrc
    VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,// CopyAccelDst
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,   //DepthRead
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,  //DepthWrite
    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,           // IndirectArgs
    VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,           //VertexRead,
    VK_ACCESS_2_INDEX_READ_BIT,                      //  IndexRead,
    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,          //RenderTarget
    VK_ACCESS_2_SHADER_READ_BIT,                     //AccelInstanceBuffer
    VK_ACCESS_2_SHADER_READ_BIT,                     // RasterRead
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, // RasterAccelRead,
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,            // RasterUAV,
};
static constexpr VkImageLayout BarrierLayoutMap[] = {
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,        // ComputeRead,
    VK_IMAGE_LAYOUT_UNDEFINED,                       // ComputeAccelRead,
    VK_IMAGE_LAYOUT_GENERAL,                         // ComputeUAV,
    VK_IMAGE_LAYOUT_GENERAL,                         // CopySource,
    VK_IMAGE_LAYOUT_GENERAL,                         // CopyDest,
    VK_IMAGE_LAYOUT_UNDEFINED,                       // BuildAccel,
    VK_IMAGE_LAYOUT_UNDEFINED,                       // CopyAccelSrc
    VK_IMAGE_LAYOUT_UNDEFINED,                       // CopyAccelDst
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, //DepthRead
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,//DepthWrite
    VK_IMAGE_LAYOUT_UNDEFINED,                       // DepthWrite
    VK_IMAGE_LAYOUT_UNDEFINED,                       //VertexRead,
    VK_IMAGE_LAYOUT_UNDEFINED,                       //  IndexRead,
    VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,              //RenderTarget
    VK_IMAGE_LAYOUT_UNDEFINED,                       //AccelInstanceBuffer
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,        // RasterRead
    VK_IMAGE_LAYOUT_UNDEFINED,                       // RasterAccelRead,
    VK_IMAGE_LAYOUT_GENERAL,                         // RasterUAV,
};

}// namespace detail
ResourceBarrier::ResourceBarrier() {}
void ResourceBarrier::add_buffer(Buffer const *buffer, size_t offset, size_t size) {
}
ResourceBarrier::~ResourceBarrier() {}
}// namespace lc::vk
