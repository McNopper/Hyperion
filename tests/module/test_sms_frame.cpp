// End-to-end exact-delta mirror: actual RT receiver producer, compute SMS trials,
// GPU continuation/resolve, and shared accumulation. Oracle replaces the planar
// mirror by a virtual area emitter; it uses no manifold/Jacobian/reciprocal code.
#include <volk/volk.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/pipeline/AccumulationPass.hpp"
#include "harmonia/presentation/ImageCapture.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/ShaderPaths.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

namespace {
struct Energy {
    double direct = 0, reflected = 0, observed = 0, mse = 0;
    std::size_t pixels = 0;
};
constexpr VkExtent2D kExtent{128, 96};
constexpr std::uint32_t kFrames = 128;
} // namespace

class SmsFrameFixture : public RtFixture {
  protected:
    void SetUp() override {
        RtFixture::SetUp();
        if (IsSkipped())
            return;
        const auto& ctx = deviceCtx();
        auto d = harmonia::renderer::Descriptors::create(ctx);
        ASSERT_TRUE(d.has_value());
        descriptors = std::make_unique<harmonia::renderer::Descriptors>(std::move(*d));
        auto p = harmonia::renderer::Pipeline::create(
            ctx, *descriptors, hyperion::makeHyperionShaderPaths(std::filesystem::path(HYPERION_SHADER_DIR)), 2);
        ASSERT_TRUE(p.has_value());
        pipeline = std::make_unique<harmonia::renderer::Pipeline>(std::move(*p));
        auto table = hyperion::ShaderBindingTable::create(ctx, *pipeline, physInfo().rtProps);
        ASSERT_TRUE(table.has_value());
        sbt = std::move(*table);
        auto receiver = harmonia::scene::Material::diffuse(sm::float3(.5F), 0).gpu();
        receiver.specularColorWeight.w = 0;
        const auto diffuse = scene.addMaterial(harmonia::scene::Material::fromGpu(receiver));
        const auto mirror = scene.addMaterial(harmonia::scene::Material::mirror(sm::float3(1)));
        const auto emitter = scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 10));
        ASSERT_NO_FATAL_FAILURE(
            quad({sm::float3(.25F, 0, -2), {4, 0, -2}, {4, 0, 2}, {.25F, 0, 2}}, {0, 1, 0}, diffuse, true));
        ASSERT_NO_FATAL_FAILURE(
            quad({sm::float3(0, 0, -2), {0, 3, -2}, {0, 3, 2}, {0, 0, 2}}, {1, 0, 0}, mirror, false));
        ASSERT_NO_FATAL_FAILURE(
            quad({sm::float3(3, 1.75F, -.25F), {3, 1.75F, .25F}, {3, 2.25F, .25F}, {3, 2.25F, -.25F}},
                 {-1, 0, 0},
                 emitter,
                 false));
        ASSERT_EQ(scene.build(ctx, commandPool()), VK_SUCCESS);
        ASSERT_EQ(scene.emissiveTriangleCount(), 2U);
        ASSERT_EQ(descriptors->updateSceneSet(ctx,
                                              scene.instanceBuffer(),
                                              scene.materialBuffer(),
                                              scene.vertexBuffer(),
                                              scene.indexBuffer(),
                                              scene.lightBuffer(),
                                              scene.emissiveTriangleBuffer(),
                                              scene.emissiveCdfBuffer(),
                                              scene.textures()),
                  VK_SUCCESS);
        auto image = harmonia::core::Image::create(ctx,
                                                   kExtent,
                                                   VK_FORMAT_R32G32B32A32_SFLOAT,
                                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                   VK_IMAGE_ASPECT_COLOR_BIT,
                                                   "test.sms.hdr");
        auto n = harmonia::core::Image::create(ctx,
                                               kExtent,
                                               VK_FORMAT_R16G16B16A16_SFLOAT,
                                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               "test.sms.normal");
        auto z = harmonia::core::Image::create(ctx,
                                               kExtent,
                                               VK_FORMAT_R32_SFLOAT,
                                               VK_IMAGE_USAGE_STORAGE_BIT,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               "test.sms.depth");
        ASSERT_TRUE(image.has_value() && n.has_value() && z.has_value());
        hdr = std::move(*image);
        normal = std::move(*n);
        depth = std::move(*z);
        hyperion::PathTracer::Config config{.samplesPerPixel = 1, .maxDepth = 3};
        auto pt = hyperion::PathTracer::create(ctx, kExtent, *pipeline, sbt, *descriptors, config);
        ASSERT_TRUE(pt.has_value()) << static_cast<int>(pt.error());
        tracer = std::move(*pt);
        auto history = harmonia::pipeline::AccumulationPass::create(
            ctx, kExtent, std::filesystem::path(HARMONIA_SHADER_DIR) / "accumulation.spv");
        ASSERT_TRUE(history.has_value());
        accumulation = std::move(*history);
        camera = harmonia::renderer::Camera(
            harmonia::renderer::Camera::Params{.position = {2, 3, 6},
                                               .target = {2, 0, 0},
                                               .up = {0, 1, 0},
                                               .vfovDeg = 43,
                                               .aspectRatio = float(kExtent.width) / kExtent.height,
                                               .nearPlane = .01F,
                                               .farPlane = 100,
                                               .lensRadius = 0,
                                               .focusDist = 6,
                                               .physical = harmonia::renderer::Camera::PhysicalCamera::fromEv100(0)});
    }

    void quad(std::array<sm::float3, 4> vertices, sm::float3 n, std::uint32_t mat, bool reverse) {
        harmonia::scene::MeshData data;
        for (auto position : vertices) {
            harmonia::GpuVertex v{};
            v.position = position;
            v.normal = n;
            data.vertices.push_back(v);
        }
        data.indices =
            reverse ? std::vector<std::uint32_t>{0, 3, 2, 0, 2, 1} : std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3};
        const auto mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(data), {}, "test.sms.quad");
        ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
        ASSERT_NE(scene.addInstance(mesh, {}, mat), std::numeric_limits<std::uint32_t>::max());
    }

    void frame(std::uint32_t index, std::uint32_t seed, std::uint64_t& continuations) {
        const auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        if (firstUse) {
            for (auto* image : {&hdr, &normal, &depth})
                image->transition(*cmd,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_PIPELINE_STAGE_2_NONE,
                                  0,
                                  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                  VK_ACCESS_2_SHADER_WRITE_BIT);
            firstUse = false;
        }
        ASSERT_EQ(tracer.render(*cmd, scene, camera, hdr, normal, depth, index), VK_SUCCESS);
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        for (std::uint32_t safety = 0;; ++safety) {
            const auto ready = tracer.frameReady();
            ASSERT_TRUE(ready.has_value()) << "frame=" << index << " seed=" << seed;
            if (*ready)
                break;
            // A test hang guard FAILS, never returns a capped/partial estimate.
            ASSERT_LT(safety, 4096U);
            const auto next = commandPool().beginOneShot();
            ASSERT_TRUE(next.has_value());
            ASSERT_EQ(tracer.recordContinuation(*next), VK_SUCCESS);
            ASSERT_EQ(commandPool().endOneShot(*next), VK_SUCCESS);
            ++continuations;
        }
        const auto post = commandPool().beginOneShot();
        ASSERT_TRUE(post.has_value());
        const harmonia::pipeline::PassContext context{.cmd = *post,
                                                      .frameIndex = index,
                                                      .extent = kExtent,
                                                      .fixedView = true,
                                                      .accumulationResetToken = seed,
                                                      .hdrBuffer = &hdr};
        accumulation.record(context);
        ASSERT_EQ(commandPool().endOneShot(*post), VK_SUCCESS);
    }

    void readPixels(std::vector<sm::float4>& result) {
        result.resize(std::size_t{kExtent.width} * kExtent.height);
        auto buffer = harmonia::core::Buffer::create(deviceCtx(),
                                                     result.size() * sizeof(sm::float4),
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                     VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                     "test.sms.pixels");
        ASSERT_TRUE(buffer.has_value());
        const auto cmd = commandPool().beginOneShot();
        ASSERT_TRUE(cmd.has_value());
        hdr.transition(*cmd,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kExtent.width, kExtent.height, 1};
        vkCmdCopyImageToBuffer(*cmd, hdr.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer->handle(), 1, &copy);
        hdr.transition(*cmd,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        VkMemoryBarrier2 memory{};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &memory;
        vkCmdPipelineBarrier2(*cmd, &dependency);
        ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
        ASSERT_EQ(buffer->readData(result.data(), result.size() * sizeof(sm::float4)), VK_SUCCESS);
    }

    Energy energy(std::span<const sm::float4> pixels) {
        Energy result;
        const double tangent = std::tan(43 * std::numbers::pi / 360);
        const double sine = 1 / std::sqrt(5.0), cosine = 2 / std::sqrt(5.0);
        constexpr int subdivisions = 24;
        for (std::uint32_t y = 0; y < kExtent.height; ++y) {
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                const double u = (2 * (x + .5) / kExtent.width - 1) * tangent * kExtent.width / kExtent.height;
                const double v = (1 - 2 * (y + .5) / kExtent.height) * tangent;
                const double dy = -sine + v * cosine, dz = -cosine - v * sine;
                const double t = -3 / dy;
                const double px = 2 + u * t, pz = 6 + dz * t;
                if (!(t > 0 && px > .6 && px < 2.7 && std::abs(pz) < 1.5))
                    continue;
                double direct = 0, reflected = 0;
                for (int iy = 0; iy < subdivisions; ++iy) {
                    for (int iz = 0; iz < subdivisions; ++iz) {
                        const double ly = 1.75 + (iy + .5) * .5 / subdivisions;
                        const double lz = -.25 + (iz + .5) * .5 / subdivisions;
                        for (int side : {-1, 1}) {
                            const double dx = side * 3 - px, dist2 = dx * dx + ly * ly + (lz - pz) * (lz - pz);
                            const double g = ly * (side * dx) / (dist2 * dist2) * .25 / (subdivisions * subdivisions);
                            if (side == 1)
                                direct += g;
                            else
                                reflected += g;
                        }
                    }
                }
                constexpr double scale = .5 / std::numbers::pi * 10 / 1.2;
                direct *= scale;
                reflected *= scale;
                const auto pixel = pixels[std::size_t{y} * kExtent.width + x];
                EXPECT_TRUE(std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z));
                const double observed = (double(pixel.x) + pixel.y + pixel.z) / 3;
                result.direct += direct;
                result.reflected += reflected;
                result.observed += observed;
                result.mse += std::pow(observed - direct - reflected, 2);
                ++result.pixels;
            }
        }
        EXPECT_GT(result.pixels, 500U);
        return result;
    }

    std::unique_ptr<harmonia::renderer::Descriptors> descriptors;
    std::unique_ptr<harmonia::renderer::Pipeline> pipeline;
    hyperion::ShaderBindingTable sbt;
    hyperion::Scene scene;
    harmonia::core::Image hdr, normal, depth;
    hyperion::PathTracer tracer;
    harmonia::pipeline::AccumulationPass accumulation;
    harmonia::renderer::Camera camera;
    bool firstUse = true;
};

TEST_F(SmsFrameFixture, MirrorMatchesVirtualEmitterEnergyAcrossSeedsAndConverges) {
    double shortMse = 0, longMse = 0;
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        hyperion::PathTracer::Config config{.samplesPerPixel = 1, .maxDepth = 3, .rngSeed = seed};
        tracer.setConfig(config);
        std::uint64_t continuations = 0;
        std::vector<sm::float4> pixels;
        for (std::uint32_t f = 0; f < kFrames; ++f) {
            ASSERT_NO_FATAL_FAILURE(frame(f, seed, continuations));
            if (f == 31) {
                ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
                shortMse += energy(pixels).mse;
            }
        }
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        const double causticRatio = (measured.observed - measured.direct) / measured.reflected;
        std::cout << "SMS mirror seed=" << seed << " caustic ratio=" << causticRatio
                  << " continuation dispatches=" << continuations << '\n';
        EXPECT_NEAR(causticRatio, 1.0, .02)
            << "Isolated reflected energy after subtracting independent direct integral";
        EXPECT_GT(continuations, kFrames) << "Must exercise actual reciprocal continuation, not a bypass";
        longMse += measured.mse;
        const auto output = std::filesystem::path(HYPERION_TEST_SHADER_DIR).parent_path() /
                            ("sms_mirror_seed" + std::to_string(seed) + ".exr");
        ASSERT_TRUE(harmonia::presentation::ImageCapture::saveExr(deviceCtx(), commandPool(), hdr, output));
    }
    std::cout << "SMS mirror aggregate MSE 128/32=" << longMse / shortMse << '\n';
    EXPECT_LT(longMse, shortMse * .65) << "Independent seeds and 4x sample budget must reduce actual image error";
}
