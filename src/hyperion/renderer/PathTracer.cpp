#include "hyperion/renderer/PathTracer.hpp"

#include <cstdint>

#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"
namespace hyperion {

// VK_EXT_ray_tracing_invocation_reorder runtime command. Not yet in the SDK volk/headers
// on all toolchains, so load it dynamically and fall back to a no-op if unavailable.
using PFN_vkCmdSetRayTracingInvocationReorderModeEXT =
    void(VKAPI_PTR*)(VkCommandBuffer commandBuffer, VkRayTracingInvocationReorderModeEXT reorderMode);

namespace {
constexpr std::uint32_t kLcgMultiplier = 1664525u;
constexpr std::uint32_t kLcgIncrement = 1013904223u;
} // namespace

std::expected<hyperion::PathTracer, VkResult> hyperion::PathTracer::create(const harmonia::DeviceContext& ctx,
                                                                           VkExtent2D renderExtent,
                                                                           const harmonia::renderer::Pipeline& pipeline,
                                                                           const hyperion::ShaderBindingTable& sbt,
                                                                           harmonia::renderer::Descriptors& descriptors,
                                                                           const Config& config) {
    auto cameraBuffer =
        harmonia::core::Buffer::create(ctx,
                                       sizeof(harmonia::CameraData),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       "hyperion.camera");
    if (!cameraBuffer) {
        return std::unexpected(cameraBuffer.error());
    }

    harmonia::core::Buffer indirectDispatchBuffer{};
    auto indirectBuf =
        harmonia::core::Buffer::create(ctx,
                                       sizeof(VkTraceRaysIndirectCommand2KHR),
                                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       "hyperion.indirectDispatch");
    if (!indirectBuf) {
        return std::unexpected(indirectBuf.error());
    }
    indirectDispatchBuffer = std::move(*indirectBuf);

    hyperion::PathTracer tracer;
    tracer.m_ctx = &ctx;
    tracer.m_rtPipeline = pipeline.rtPipeline();
    tracer.m_pipelineLayout = descriptors.pipelineLayout();
    tracer.m_descriptors = &descriptors; // MOD1: bindSceneSet() replaces set1()
    tracer.m_extent = renderExtent;
    tracer.m_config = config;
    tracer.m_cameraBuffer = std::move(*cameraBuffer);
    tracer.m_indirectDispatchBuffer = std::move(indirectDispatchBuffer);
    tracer.m_raygen = sbt.raygenRegion();
    tracer.m_miss = sbt.missRegion();
    tracer.m_hit = sbt.hitRegion();
    tracer.updateIndirectBuffer();
    return tracer;
}

VkResult hyperion::PathTracer::render(VkCommandBuffer cmd,
                                      const hyperion::Scene& scene,
                                      const harmonia::renderer::Camera& camera,
                                      const harmonia::core::Image& hdrImage,
                                      const harmonia::core::Image& gNormal,
                                      const harmonia::core::Image& gDepth,
                                      std::uint32_t frameIndex) noexcept {
    if (cmd == VK_NULL_HANDLE || scene.tlas() == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    writeFrameDescriptors(cmd, scene, camera, frameIndex, hdrImage, gNormal, gDepth);
    pushFrameConstants(cmd, scene, frameIndex);

    if (m_config.serEnabled) {
        auto* setMode = reinterpret_cast<PFN_vkCmdSetRayTracingInvocationReorderModeEXT>(
            vkGetDeviceProcAddr(m_ctx->device, "vkCmdSetRayTracingInvocationReorderModeEXT"));
        if (setMode != nullptr) {
            setMode(cmd, VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT);
        }
    }

    dispatchRays(cmd);

    return VK_SUCCESS;
}

void hyperion::PathTracer::writeFrameDescriptors(VkCommandBuffer cmd,
                                                 const hyperion::Scene& scene,
                                                 const harmonia::renderer::Camera& camera,
                                                 std::uint32_t frameIndex,
                                                 const harmonia::core::Image& hdrImage,
                                                 const harmonia::core::Image& gNormal,
                                                 const harmonia::core::Image& gDepth) noexcept {
    const harmonia::CameraData cameraData = camera.getCameraData(frameIndex, m_config.maxDepth);
    m_cameraBuffer.uploadData(&cameraData, sizeof(cameraData), 0);

    const VkAccelerationStructureKHR tlasHandle = scene.tlas();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    // MOD1: write set 0 via descriptor buffer + bind both sets.
    m_descriptors->updateFrameSet(*m_ctx, tlasHandle, hdrImage.view(), m_cameraBuffer, gNormal.view(), gDepth.view());
    m_descriptors->bindSceneSet(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);
}

void hyperion::PathTracer::pushFrameConstants(VkCommandBuffer cmd,
                                              const hyperion::Scene& scene,
                                              std::uint32_t frameIndex) noexcept {
    const harmonia::PushConstants pushConstants{
        .frameIndex = frameIndex,
        .maxDepth = m_config.maxDepth,
        .rngSeed = frameIndex * kLcgMultiplier + kLcgIncrement,
        .envLuminanceScale = m_config.envLuminance,
        .lightCount = scene.lightCount(),
        .outputColorSpace = static_cast<std::uint32_t>(m_config.outputColorSpace),
        .samplesPerPixel = m_config.samplesPerPixel,
        .hasEnvMap = m_config.hasEnvMap,
        .emissiveTriangleCount = scene.emissiveTriangleCount(),
        .envImportanceWidth = m_config.envImportanceWidth,
        .envImportanceHeight = m_config.envImportanceHeight,
        .tonemapper = static_cast<std::uint32_t>(m_config.tonemapper),
        .workingColorSpace = static_cast<std::uint32_t>(m_config.workingColorSpace),
    };
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(harmonia::PushConstants), &pushConstants);
}

void hyperion::PathTracer::dispatchRays(VkCommandBuffer cmd) noexcept {
    static auto pfnTraceRaysIndirect2 = reinterpret_cast<PFN_vkCmdTraceRaysIndirect2KHR>(
        vkGetDeviceProcAddr(m_ctx->device, "vkCmdTraceRaysIndirect2KHR"));

    pfnTraceRaysIndirect2(cmd, m_indirectDispatchBuffer.deviceAddress());
}

void hyperion::PathTracer::setConfig(const Config& config) noexcept {
    m_config = config;
}

void hyperion::PathTracer::onResize(VkExtent2D newExtent) noexcept {
    m_extent = newExtent;
    updateIndirectBuffer();
}

void hyperion::PathTracer::updateIndirectBuffer() noexcept {
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
}

} // namespace hyperion
