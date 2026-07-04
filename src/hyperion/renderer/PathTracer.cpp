#include "hyperion/renderer/PathTracer.hpp"

#include <array>

#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

// VK_EXT_ray_tracing_invocation_reorder runtime command. Not yet in the SDK volk/headers
// on all toolchains, so load it dynamically and fall back to a no-op if unavailable.
using PFN_vkCmdSetRayTracingInvocationReorderModeEXT = void(VKAPI_PTR*)(VkCommandBuffer commandBuffer,
                                                                        VkRayTracingInvocationReorderModeEXT reorderMode);

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
                                                       const Config& config) {
    auto cameraBuffer = Buffer::create(ctx,
                                       sizeof(CameraData),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       "hyperion.camera");
    if (!cameraBuffer) {
        return std::unexpected(cameraBuffer.error());
    }

    Buffer indirectDispatchBuffer{};
    if (config.indirectRt2Enabled) {
        auto indirectBuf = Buffer::create(ctx,
                                          sizeof(VkTraceRaysIndirectCommand2KHR),
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                          VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          "hyperion.indirectDispatch");
        if (!indirectBuf) {
            return std::unexpected(indirectBuf.error());
        }
        indirectDispatchBuffer = std::move(*indirectBuf);
    }

    PathTracer tracer;
    tracer.m_ctx = &ctx;
    tracer.m_rtPipeline = pipeline.rtPipeline();
    tracer.m_pipelineLayout = descriptors.pipelineLayout();
    tracer.m_sceneSet = descriptors.set1();
    tracer.m_extent = renderExtent;
    tracer.m_config = config;
    tracer.m_cameraBuffer = std::move(*cameraBuffer);
    tracer.m_indirectDispatchBuffer = std::move(indirectDispatchBuffer);
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
        .workingColorSpace = m_config.workingColorSpace,
    };
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pushConstants);

    if (m_config.serEnabled) {
        auto* setMode = reinterpret_cast<PFN_vkCmdSetRayTracingInvocationReorderModeEXT>(
            vkGetDeviceProcAddr(m_ctx->device, "vkCmdSetRayTracingInvocationReorderModeEXT"));
        if (setMode != nullptr) {
            setMode(cmd, VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT);
        }
    }

    if (m_config.indirectRt2Enabled && m_indirectDispatchBuffer.isValid()) {
        static auto pfnTraceRaysIndirect2 = reinterpret_cast<PFN_vkCmdTraceRaysIndirect2KHR>(
            vkGetDeviceProcAddr(m_ctx->device, "vkCmdTraceRaysIndirect2KHR"));

        if (pfnTraceRaysIndirect2 != nullptr) {
            VkTraceRaysIndirectCommand2KHR indirectCmd{};
            indirectCmd.raygenShaderRecordAddress = m_raygen.deviceAddress;
            indirectCmd.raygenShaderRecordSize = m_raygen.size;
            indirectCmd.missShaderBindingTableAddress = m_miss.deviceAddress;
            indirectCmd.missShaderBindingTableSize = m_miss.size;
            indirectCmd.missShaderBindingTableStride = m_miss.stride;
            indirectCmd.hitShaderBindingTableAddress = m_hit.deviceAddress;
            indirectCmd.hitShaderBindingTableSize = m_hit.size;
            indirectCmd.hitShaderBindingTableStride = m_hit.stride;
            indirectCmd.callableShaderBindingTableAddress = m_callable.deviceAddress;
            indirectCmd.callableShaderBindingTableSize = m_callable.size;
            indirectCmd.callableShaderBindingTableStride = m_callable.stride;
            indirectCmd.width = m_extent.width;
            indirectCmd.height = m_extent.height;
            indirectCmd.depth = 1;
            m_indirectDispatchBuffer.uploadData(&indirectCmd, sizeof(indirectCmd), 0);

            const VkMemoryBarrier2 hostToIndirect{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .pNext = nullptr,
                .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
            };
            const VkDependencyInfo dep{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = 0,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &hostToIndirect,
                .bufferMemoryBarrierCount = 0,
                .pBufferMemoryBarriers = nullptr,
                .imageMemoryBarrierCount = 0,
                .pImageMemoryBarriers = nullptr,
            };
            vkCmdPipelineBarrier2(cmd, &dep);

            pfnTraceRaysIndirect2(cmd, m_indirectDispatchBuffer.deviceAddress());
        } else {
            vkCmdTraceRaysKHR(cmd, &m_raygen, &m_miss, &m_hit, &m_callable, m_extent.width, m_extent.height, 1);
        }
    } else {
        vkCmdTraceRaysKHR(cmd, &m_raygen, &m_miss, &m_hit, &m_callable, m_extent.width, m_extent.height, 1);
    }

    return VK_SUCCESS;
}

void PathTracer::setConfig(const Config& config) noexcept {
    m_config = config;
}

void PathTracer::onResize(VkExtent2D newExtent) noexcept {
    m_extent = newExtent;
}
