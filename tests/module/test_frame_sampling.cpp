// Actual ray-tracing shader + shared host accumulation, with deterministic
// emitter radiances so history contamination cannot hide inside Monte Carlo noise.
#include <volk/volk.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <utility>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/pipeline/AccumulationPass.hpp"
#include "harmonia/pipeline/PassContext.hpp"
#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/ShaderPaths.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

namespace {
struct Readback {
    VmaAllocator allocator{};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    void* mapped{};
    Readback() = default;
    Readback(const Readback&) = delete;
    Readback& operator=(const Readback&) = delete;
    ~Readback() {
        if (buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, buffer, allocation);
    }
};
} // namespace

class FrameSamplingFixture : public RtFixture {
  protected:
    void SetUp() override {
        RtFixture::SetUp();
        if (IsSkipped())
            return;
        auto descriptors = harmonia::renderer::Descriptors::create(deviceCtx());
        ASSERT_TRUE(descriptors.has_value());
        m_descriptors = std::make_unique<harmonia::renderer::Descriptors>(std::move(*descriptors));
        auto pipeline = harmonia::renderer::Pipeline::create(
            deviceCtx(),
            *m_descriptors,
            hyperion::makeHyperionShaderPaths(std::filesystem::path(HYPERION_SHADER_DIR)),
            2U);
        ASSERT_TRUE(pipeline.has_value());
        m_pipeline = std::make_unique<harmonia::renderer::Pipeline>(std::move(*pipeline));
        auto sbt = hyperion::ShaderBindingTable::create(deviceCtx(), *m_pipeline, physInfo().rtProps);
        ASSERT_TRUE(sbt.has_value());
        m_sbt = std::make_unique<hyperion::ShaderBindingTable>(std::move(*sbt));

        const auto material = m_scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 0.25F));
        harmonia::scene::MeshData data;
        for (auto xy : {sm::float2(-10, -10), sm::float2(10, -10), sm::float2(10, 10), sm::float2(-10, 10)}) {
            harmonia::GpuVertex v{};
            v.position = {xy.x, xy.y, 0};
            v.normal = {0, 0, 1};
            data.vertices.push_back(v);
        }
        data.indices = {0, 1, 2, 0, 2, 3};
        const auto mesh = m_scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.emitter");
        ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
        ASSERT_NE(m_scene.addInstance(mesh, {}, material), std::numeric_limits<std::uint32_t>::max());
        ASSERT_EQ(m_scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
        ASSERT_EQ(m_descriptors->updateSceneSet(deviceCtx(),
                                                m_scene.instanceBuffer(),
                                                m_scene.materialBuffer(),
                                                m_scene.vertexBuffer(),
                                                m_scene.indexBuffer(),
                                                m_scene.lightBuffer(),
                                                m_scene.emissiveTriangleBuffer(),
                                                m_scene.emissiveCdfBuffer(),
                                                m_scene.textures()),
                  VK_SUCCESS);

        auto hdr = harmonia::core::Image::create(deviceCtx(),
                                                 kExtent,
                                                 VK_FORMAT_R32G32B32A32_SFLOAT,
                                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 "test.frame.hdr");
        auto normal = harmonia::core::Image::create(deviceCtx(),
                                                    kExtent,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    VK_IMAGE_USAGE_STORAGE_BIT,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    "test.frame.normal");
        auto depth = harmonia::core::Image::create(deviceCtx(),
                                                   kExtent,
                                                   VK_FORMAT_R32_SFLOAT,
                                                   VK_IMAGE_USAGE_STORAGE_BIT,
                                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                                   "test.frame.depth");
        ASSERT_TRUE(hdr.has_value() && normal.has_value() && depth.has_value());
        m_hdr = std::move(*hdr);
        m_normal = std::move(*normal);
        m_depth = std::move(*depth);
        auto tracer = hyperion::PathTracer::create(deviceCtx(),
                                                   kExtent,
                                                   *m_pipeline,
                                                   *m_sbt,
                                                   *m_descriptors,
                                                   hyperion::PathTracer::Config{.samplesPerPixel = 1U, .maxDepth = 1U});
        ASSERT_TRUE(tracer.has_value());
        m_tracer = std::move(*tracer);
        auto accumulation = harmonia::pipeline::AccumulationPass::create(
            deviceCtx(), kExtent, std::filesystem::path(HARMONIA_SHADER_DIR) / "accumulation.spv");
        ASSERT_TRUE(accumulation.has_value());
        m_accumulation = std::move(*accumulation);
        m_camera = harmonia::renderer::Camera(harmonia::renderer::Camera::Params{
            .position = {0, 0, 2},
            .target = {0, 0, 0},
            .up = {0, 1, 0},
            .vfovDeg = 45,
            .aspectRatio = 1,
            .nearPlane = 0.1F,
            .farPlane = 100,
            .lensRadius = 0,
            .focusDist = 2,
            .physical = harmonia::renderer::Camera::PhysicalCamera{.aperture = 1, .shutterSpeedHz = 1, .iso = 100}});
        m_readback.allocator = deviceCtx().allocator;
        VkBufferCreateInfo buffer{};
        buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer.size = sizeof(sm::float4);
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocation.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        VmaAllocationInfo mapped{};
        ASSERT_EQ(vmaCreateBuffer(
                      deviceCtx().allocator, &buffer, &allocation, &m_readback.buffer, &m_readback.allocation, &mapped),
                  VK_SUCCESS);
        m_readback.mapped = mapped.pMappedData;
        ASSERT_NE(m_readback.mapped, nullptr);
    }

    void frame(float emission, std::uint32_t index, bool accumulate) {
        auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        if (index == 0) {
            for (auto* image : {&m_hdr, &m_normal, &m_depth})
                image->transition(*cmd,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_NONE,
                                  0,
                                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                  VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        }
        auto material = harmonia::scene::Material::emissive(sm::float3(1), emission).gpu();
        vkCmdUpdateBuffer(*cmd, m_scene.materialBuffer().handle(), 0, sizeof(material), &material);
        const VkMemoryBarrier2 updated{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                       .pNext = nullptr,
                                       .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                       .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                       .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT};
        const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                          .pNext = nullptr,
                                          .dependencyFlags = 0,
                                          .memoryBarrierCount = 1,
                                          .pMemoryBarriers = &updated,
                                          .bufferMemoryBarrierCount = 0,
                                          .pBufferMemoryBarriers = nullptr,
                                          .imageMemoryBarrierCount = 0,
                                          .pImageMemoryBarriers = nullptr};
        vkCmdPipelineBarrier2(*cmd, &dependency);
        ASSERT_EQ(m_tracer.render(*cmd, m_scene, m_camera, m_hdr, m_normal, m_depth, index), VK_SUCCESS);
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        const auto ready = m_tracer.frameReady();
        ASSERT_TRUE(ready.has_value() && *ready) << "Primary-emitter depth-1 frames must complete immediately";
        cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        if (accumulate) {
            const harmonia::pipeline::PassContext context{.cmd = *cmd,
                                                          .frameIndex = index,
                                                          .extent = kExtent,
                                                          .fixedView = true,
                                                          .accumulationResetToken = 1,
                                                          .hdrBuffer = &m_hdr};
            m_accumulation.record(context);
        }
        m_hdr.transition(*cmd,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT);
        const VkBufferImageCopy region{.bufferOffset = 0,
                                       .bufferRowLength = 0,
                                       .bufferImageHeight = 0,
                                       .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                       .imageOffset = {0, 0, 0},
                                       .imageExtent = {1, 1, 1}};
        vkCmdCopyImageToBuffer(
            *cmd, m_hdr.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_readback.buffer, 1, &region);
        m_hdr.transition(*cmd,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        ASSERT_EQ(vmaInvalidateAllocation(deviceCtx().allocator, m_readback.allocation, 0, VK_WHOLE_SIZE), VK_SUCCESS);
        const auto pixel = *static_cast<const sm::float4*>(m_readback.mapped);
        m_last = {pixel.x, pixel.y, pixel.z};
    }
    void expectColor(float value) {
        for (float channel : m_last)
            EXPECT_NEAR(channel, value * m_camera.physicalCamera().exposure(), 1.0e-6F);
    }

    static constexpr VkExtent2D kExtent{8, 8};
    std::unique_ptr<harmonia::renderer::Descriptors> m_descriptors;
    std::unique_ptr<harmonia::renderer::Pipeline> m_pipeline;
    std::unique_ptr<hyperion::ShaderBindingTable> m_sbt;
    hyperion::Scene m_scene;
    hyperion::PathTracer m_tracer;
    harmonia::renderer::Camera m_camera;
    harmonia::core::Image m_hdr, m_normal, m_depth;
    harmonia::pipeline::AccumulationPass m_accumulation;
    Readback m_readback;
    std::array<float, 3> m_last{};
};

TEST_F(FrameSamplingFixture, RaygenWritesOnlyTheCurrentFrame) {
    ASSERT_NO_FATAL_FAILURE(frame(0.25F, 0, false));
    expectColor(0.25F);
    ASSERT_NO_FATAL_FAILURE(frame(0.75F, 1, false));
    expectColor(0.75F);
}

TEST_F(FrameSamplingFixture, SharedAccumulationIsTheOnlyHistoryOwner) {
    ASSERT_NO_FATAL_FAILURE(frame(0.25F, 0, true));
    expectColor(0.25F);
    ASSERT_NO_FATAL_FAILURE(frame(0.75F, 1, true));
    expectColor(0.5F);
}
