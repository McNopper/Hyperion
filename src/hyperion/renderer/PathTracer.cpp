#include "hyperion/renderer/PathTracer.hpp"

#include <array>

#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "harmonia/scene/Scene.hpp"

std::expected<PathTracer, VkResult> PathTracer::create(const DeviceContext& ctx,
                                                       VkExtent2D renderExtent,
                                                       const Pipeline& pipeline,
                                                       const ShaderBindingTable& sbt,
                                                       const Descriptors& descriptors) {
    return create(ctx, renderExtent, pipeline, sbt, descriptors, Config{});
}

std::expected<PathTracer, VkResult> PathTracer::create(const DeviceContext& ctx,
                                                       VkExtent2D renderExtent,
                                                       const Pipeline& pipeline,
                                                       const ShaderBindingTable& sbt,
                                                       const Descriptors& descriptors,
                                                       Config config) {
    auto cameraBuffer = Buffer::create(ctx,
                                       sizeof(CameraData),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       "hyperion.camera");
    if (!cameraBuffer) {
        return std::unexpected(cameraBuffer.error());
    }

    PathTracer tracer;
    tracer.m_rtPipeline = pipeline.rtPipeline();
    tracer.m_pipelineLayout = descriptors.pipelineLayout();
    tracer.m_sceneSet = descriptors.set1();
    tracer.m_extent = renderExtent;
    tracer.m_config = config;
    tracer.m_cameraBuffer = std::move(*cameraBuffer);
    tracer.m_raygen = sbt.raygenRegion();
    tracer.m_miss = sbt.missRegion();
    tracer.m_hit = sbt.hitRegion();
    tracer.m_callable = sbt.callableRegion();
    return tracer;
}

VkResult PathTracer::render(VkCommandBuffer cmd,
                            const Scene& scene,
                            const Camera& camera,
                            const Image& hdrImage,
                            const Image& gNormal,
                            const Image& gDepth,
                            uint32_t frameIndex) noexcept {
    if (cmd == VK_NULL_HANDLE || scene.tlas() == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const CameraData cameraData = camera.getCameraData(frameIndex, m_config.maxDepth);
    m_cameraBuffer.uploadData(&cameraData, sizeof(cameraData), 0);

    const VkDescriptorImageInfo hdrInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = hdrImage.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorBufferInfo cameraInfo{
        .buffer = m_cameraBuffer.handle(),
        .offset = 0,
        .range = sizeof(CameraData),
    };
    const VkDescriptorImageInfo gNormalInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = gNormal.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo gDepthInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = gDepth.view(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkAccelerationStructureKHR tlasHandle = scene.tlas();
    const VkWriteDescriptorSetAccelerationStructureKHR asInfo{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .pNext = nullptr,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlasHandle,
    };

    // Push bindings 0–2 (tlas, hdrTarget, camera) and 4–5 (gNormal, gDepth).
    // Binding 3 (swapchainOutput) is pushed by ToneMapper when it runs.
    std::array<VkWriteDescriptorSet, 5> writes{};
    writes[0] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = &asInfo,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        .pImageInfo = nullptr,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    writes[1] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &hdrInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    writes[2] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &cameraInfo,
        .pTexelBufferView = nullptr,
    };
    writes[3] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 4,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &gNormalInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    writes[4] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = VK_NULL_HANDLE,
        .dstBinding = 5,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &gDepthInfo,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    vkCmdPushDescriptorSet(cmd,
                           VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                           m_pipelineLayout,
                           0,
                           static_cast<uint32_t>(writes.size()),
                           writes.data());
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 1, 1, &m_sceneSet, 0, nullptr);

    const PushConstants pushConstants{
        .frameIndex = frameIndex,
        .maxDepth = m_config.maxDepth,
        .rngSeed = frameIndex * 1664525u + 1013904223u,
        .envLuminanceScale = m_config.envLuminance,
        .lightCount = scene.lightCount(),
        .outputColorSpace = m_config.outputColorSpace,
        .samplesPerPixel = m_config.samplesPerPixel,
        .hasEnvMap = m_config.hasEnvMap,
        .emissiveTriangleCount = scene.emissiveTriangleCount(),
        .envImportanceWidth = m_config.envImportanceWidth,
        .envImportanceHeight = m_config.envImportanceHeight,
        .tonemapper = m_config.tonemapper,
    };
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pushConstants);

    vkCmdTraceRaysKHR(cmd, &m_raygen, &m_miss, &m_hit, &m_callable, m_extent.width, m_extent.height, 1);
    return VK_SUCCESS;
}

void PathTracer::setConfig(Config config) noexcept {
    m_config = config;
}

void PathTracer::onResize(VkExtent2D newExtent) noexcept {
    m_extent = newExtent;
}
