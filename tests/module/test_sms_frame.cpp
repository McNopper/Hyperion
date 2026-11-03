// End-to-end exact-delta mirror: actual RT receiver producer, compute SMS trials,
// GPU continuation/resolve, and shared accumulation. Oracle replaces the planar
// mirror by a virtual area emitter; it uses no manifold/Jacobian/reciprocal code.
#include <volk/volk.h>

#include <algorithm>
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
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "hyperion/ShaderPaths.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

namespace {
struct Energy {
    double direct = 0, reflected = 0, observed = 0, mse = 0;
    std::size_t pixels = 0;
};
struct AbsorptionEnergy {
    std::array<double, 3> expected{}, observed{};
    double mse = 0;
    std::size_t pixels = 0;
};
struct AbsorptionPrediction {
    std::size_t pixel;
    std::array<double, 3> value;
};
constexpr VkExtent2D kExtent{128, 96};
constexpr std::uint32_t kFrames = 128;
} // namespace

class SmsFrameFixture : public RtFixture {
  protected:
    [[nodiscard]] virtual bool glassSlab() const noexcept { return false; }
    [[nodiscard]] virtual float absorptionDepth() const noexcept { return 0; }
    [[nodiscard]] virtual std::uint32_t partitionResolution() const noexcept { return 1; }
    [[nodiscard]] virtual bool priorEnabled() const noexcept { return false; }
    [[nodiscard]] virtual bool specularPrefix() const noexcept { return false; }
    [[nodiscard]] virtual bool slabPrefix() const noexcept { return false; }
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
        auto casterMaterial =
            glassSlab() ? harmonia::scene::Material::glass(1.5F, 0) : harmonia::scene::Material::mirror(sm::float3(1));
        if (absorptionDepth() > 0) {
            auto gpu = casterMaterial.gpu();
            gpu.transmissionParams.x = absorptionDepth();
            gpu.transmissionColorWeight = {.6F, .8F, .9F, 1};
            gpu.transmissionScatter = sm::float4(0);
            casterMaterial = harmonia::scene::Material::fromGpu(gpu);
        }
        const auto caster = scene.addMaterial(std::move(casterMaterial));
        const auto emitter = scene.addMaterial(harmonia::scene::Material::emissive(sm::float3(1), 10));
        if (specularPrefix()) {
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(.25F, 0, 4.25F), {4, 0, 4.25F}, {4, 0, 8}, {.25F, 0, 8}}, {0, 1, 0}, diffuse, true));
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(0, 0, 4), {0, 3, 4}, {0, 3, 8}, {0, 0, 8}}, {1, 0, 0}, caster, false));
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(3, 1.75F, 5.75F), {3, 1.75F, 6.25F}, {3, 2.25F, 6.25F}, {3, 2.25F, 5.75F}},
                     {-1, 0, 0},
                     emitter,
                     false));
            auto gray = harmonia::scene::Material::mirror(sm::float3(.4F)).gpu();
            gray.specularColorWeight = {0, 0, 0, 1};
            const auto prefix = scene.addMaterial(harmonia::scene::Material::fromGpu(gray));
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(.25F, 0, 4), {4, 0, 4}, {4, 4, 4}, {.25F, 4, 4}}, {0, 0, 1}, prefix, false));
        } else {
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(.25F, 0, -2), {4, 0, -2}, {4, 0, 2}, {.25F, 0, 2}}, {0, 1, 0}, diffuse, true));
        }
        if (glassSlab()) {
            const auto slab = scene.addMesh(ctx,
                                            commandPool(),
                                            harmonia::scene::ProceduralGeometry::makeBox({6, .25F, 5}),
                                            {},
                                            "test.sms.glass.slab");
            ASSERT_NE(slab, std::numeric_limits<std::uint32_t>::max());
            ASSERT_NE(scene.addInstance(slab, harmonia::scene::Transform{.translation = {2, 1.25F, 0}}, caster),
                      std::numeric_limits<std::uint32_t>::max());
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(1.75F, 3, -.25F), {2.25F, 3, -.25F}, {2.25F, 3, .25F}, {1.75F, 3, .25F}},
                     {0, -1, 0},
                     emitter,
                     false));
            if (slabPrefix()) {
                auto gray = harmonia::scene::Material::mirror(sm::float3(.4F)).gpu();
                gray.specularColorWeight = {0, 0, 0, 1};
                const auto prefix = scene.addMaterial(harmonia::scene::Material::fromGpu(gray));
                // Reflect camera (2,.5,1) in z=2 to the Snell oracle's (2,.5,3).
                // This narrow mirror covers camera rays but not receiver-to-light R tails.
                ASSERT_NO_FATAL_FAILURE(quad({sm::float3(.25F, .05F, 2), {.25F, .45F, 2}, {4, .45F, 2}, {4, .05F, 2}},
                                             {0, 0, -1},
                                             prefix,
                                             false));
            }
        } else if (!specularPrefix()) {
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(0, 0, -2), {0, 3, -2}, {0, 3, 2}, {0, 0, 2}}, {1, 0, 0}, caster, false));
            ASSERT_NO_FATAL_FAILURE(
                quad({sm::float3(3, 1.75F, -.25F), {3, 1.75F, .25F}, {3, 2.25F, .25F}, {3, 2.25F, -.25F}},
                     {-1, 0, 0},
                     emitter,
                     false));
        }
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
        hyperion::PathTracer::Config config{.samplesPerPixel = 1,
                                            .maxDepth = glassSlab() || specularPrefix() ? 4U : 3U,
                                            .smsPartitionResolution = partitionResolution(),
                                            .smsPriorEnabled = priorEnabled()};
        auto pt = hyperion::PathTracer::create(ctx, kExtent, *pipeline, sbt, *descriptors, config);
        ASSERT_TRUE(pt.has_value()) << static_cast<int>(pt.error());
        tracer = std::move(*pt);
        auto history = harmonia::pipeline::AccumulationPass::create(
            ctx, kExtent, std::filesystem::path(HARMONIA_SHADER_DIR) / "accumulation.spv");
        ASSERT_TRUE(history.has_value());
        accumulation = std::move(*history);
        auto cameraParams =
            harmonia::renderer::Camera::Params{.position = {2, 3, 6},
                                               .target = {2, 0, 0},
                                               .up = {0, 1, 0},
                                               .vfovDeg = 43,
                                               .aspectRatio = float(kExtent.width) / kExtent.height,
                                               .nearPlane = .01F,
                                               .farPlane = 100,
                                               .lensRadius = 0,
                                               .focusDist = 6,
                                               .physical = harmonia::renderer::Camera::PhysicalCamera::fromEv100(0)};
        if (glassSlab()) {
            cameraParams.position = {2, .5F, 3};
            cameraParams.vfovDeg = 10;
            cameraParams.focusDist = 3;
        }
        if (specularPrefix())
            cameraParams.target = {2, 0, 2};
        if (slabPrefix()) {
            cameraParams.position = {2, .5F, 1};
            cameraParams.target = {2, 0, 4};
        }
        camera = harmonia::renderer::Camera(cameraParams);
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

    Energy slabEnergy(std::span<const sm::float4> pixels) {
        if (slabPredictions.empty()) {
            // Independent radial Snell mapping for two PARALLEL interfaces.
            // Air height a=1+1.5, glass thickness d=.5, eta=1.5. With t=tan(theta),
            // rho=a*t+d*t/sqrt(eta^2+(eta^2-1)*t^2); dOmega/dA follows the radial
            // area Jacobian. No renderer manifold or reciprocal code is called.
            constexpr std::array nodes{-0.9602898564975363,
                                       -0.7966664774136267,
                                       -0.5255324099163290,
                                       -0.1834346424956498,
                                       0.1834346424956498,
                                       0.5255324099163290,
                                       0.7966664774136267,
                                       0.9602898564975363};
            constexpr std::array weights{0.1012285362903763,
                                         0.2223810344533745,
                                         0.3137066458778873,
                                         0.3626837833783620,
                                         0.3626837833783620,
                                         0.3137066458778873,
                                         0.2223810344533745,
                                         0.1012285362903763};
            constexpr double a = 2.5, d = .5, eta = 1.5;
            const double tangent = std::tan(10 * std::numbers::pi / 360);
            const double sine = .5 / std::sqrt(9.25), cosine = 3 / std::sqrt(9.25);
            double maximumResidual = 0;
            for (std::uint32_t y = 0; y < kExtent.height; ++y)
                for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                    const double u = (2 * (x + .5) / kExtent.width - 1) * tangent * kExtent.width / kExtent.height;
                    const double v = (1 - 2 * (y + .5) / kExtent.height) * tangent;
                    const double rayY = -sine + v * cosine, rayZ = -cosine - v * sine, distance = -.5 / rayY;
                    const double px = 2 + u * distance, pz = 3 + rayZ * distance;
                    if (!(distance > 0 && px > .6 && px < 2.7 && std::abs(pz) < 1.5))
                        continue;
                    double integral = 0;
                    for (std::size_t ix = 0; ix < nodes.size(); ++ix)
                        for (std::size_t iz = 0; iz < nodes.size(); ++iz) {
                            const double lx = 2 + .25 * nodes[ix], lz = .25 * nodes[iz],
                                         rho = std::hypot(lx - px, lz - pz);
                            double t = rho / (a + d / eta);
                            for (int iteration = 0; iteration < 32; ++iteration) {
                                const double D = std::sqrt(eta * eta + (eta * eta - 1) * t * t);
                                const double residual = a * t + d * t / D - rho;
                                const double derivative = a + d * eta * eta / (D * D * D);
                                const double step = residual / derivative;
                                t -= step;
                                if (std::abs(step) < 1e-14)
                                    break;
                            }
                            const double D = std::sqrt(eta * eta + (eta * eta - 1) * t * t);
                            maximumResidual = std::max(maximumResidual, std::abs(a * t + d * t / D - rho));
                            const double derivative = a + d * eta * eta / (D * D * D);
                            const double geometric = rho == 0 ? 1 / std::pow(a + d / eta, 2)
                                                              : t / (rho * std::pow(1 + t * t, 1.5) * derivative);
                            const double cosAir = 1 / std::sqrt(1 + t * t);
                            const double cosGlass = std::sqrt(1 - (1 - cosAir * cosAir) / (eta * eta));
                            const double rs = (cosAir - eta * cosGlass) / (cosAir + eta * cosGlass);
                            const double rp = (eta * cosAir - cosGlass) / (eta * cosAir + cosGlass);
                            const double fresnel = .5 * (rs * rs + rp * rp);
                            integral +=
                                weights[ix] * weights[iz] * .0625 * cosAir * geometric * (1 - fresnel) * (1 - fresnel);
                        }
                    slabPredictions.push_back(
                        {std::size_t{y} * kExtent.width + x, integral * .5 / std::numbers::pi * 10 / 1.2});
                }
            EXPECT_LE(maximumResidual, 1e-12) << "Independent scalar Snell oracle must converge";
        }
        Energy result;
        for (auto [index, expected] : slabPredictions) {
            const auto pixel = pixels[index];
            EXPECT_TRUE(std::isfinite(pixel.x) && std::isfinite(pixel.y) && std::isfinite(pixel.z));
            const double observed = (double(pixel.x) + pixel.y + pixel.z) / 3;
            result.observed += observed;
            result.reflected += expected; // manifold energy; direct NEE is blocked by the slab
            result.mse += (observed - expected) * (observed - expected);
            ++result.pixels;
        }
        EXPECT_GT(result.pixels, 500U);
        return result;
    }

    AbsorptionEnergy absorptionEnergy(std::span<const sm::float4> pixels) {
        if (absorptionPredictions.empty()) {
            constexpr std::array nodes{-0.9602898564975363,
                                       -0.7966664774136267,
                                       -0.5255324099163290,
                                       -0.1834346424956498,
                                       0.1834346424956498,
                                       0.5255324099163290,
                                       0.7966664774136267,
                                       0.9602898564975363};
            constexpr std::array weights{0.1012285362903763,
                                         0.2223810344533745,
                                         0.3137066458778873,
                                         0.3626837833783620,
                                         0.3626837833783620,
                                         0.3137066458778873,
                                         0.2223810344533745,
                                         0.1012285362903763};
            constexpr std::array colors{.6F, .8F, .9F};
            constexpr double a = 2.5, d = .5, eta = 1.5, scale = .5 / std::numbers::pi * 10 / 1.2;
            const double tangent = std::tan(10 * std::numbers::pi / 360);
            const double sine = .5 / std::sqrt(9.25), cosine = 3 / std::sqrt(9.25);
            double maximumResidual = 0, maximumFresnel = 0, minimumPrediction = 1e30;
            for (std::uint32_t y = 0; y < kExtent.height; ++y)
                for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                    const double u = (2 * (x + .5) / kExtent.width - 1) * tangent * kExtent.width / kExtent.height;
                    const double v = (1 - 2 * (y + .5) / kExtent.height) * tangent;
                    const double distance = -.5 / (-sine + v * cosine), px = 2 + u * distance,
                                 pz = 3 + (-cosine - v * sine) * distance;
                    if (!(distance > 0 && px > .6 && px < 2.7 && std::abs(pz) < 1.5))
                        continue;
                    std::array<double, 3> prediction{};
                    for (std::size_t ix = 0; ix < nodes.size(); ++ix)
                        for (std::size_t iz = 0; iz < nodes.size(); ++iz) {
                            const double rho = std::hypot(2 + .25 * nodes[ix] - px, .25 * nodes[iz] - pz);
                            // Bulk PT does not charge internal boundary reflections to the
                            // surface budget. Include even-reflection upper exits as well as
                            // the SMS-owned first TT crossing, by unfolding the slab.
                            for (int pairs = 0; pairs < 4; ++pairs) {
                                const double unfoldedDepth = (2 * pairs + 1) * d;
                                double t = rho / (a + unfoldedDepth / eta);
                                for (int iteration = 0; iteration < 32; ++iteration) {
                                    const double D = std::sqrt(eta * eta + (eta * eta - 1) * t * t);
                                    const double step = (a * t + unfoldedDepth * t / D - rho) /
                                                        (a + unfoldedDepth * eta * eta / (D * D * D));
                                    t -= step;
                                    if (std::abs(step) < 1e-14)
                                        break;
                                }
                                const double D = std::sqrt(eta * eta + (eta * eta - 1) * t * t);
                                maximumResidual =
                                    std::max(maximumResidual, std::abs(a * t + unfoldedDepth * t / D - rho));
                                const double derivative = a + unfoldedDepth * eta * eta / (D * D * D);
                                const double geometric = rho == 0 ? 1 / std::pow(a + unfoldedDepth / eta, 2)
                                                                  : t / (rho * std::pow(1 + t * t, 1.5) * derivative);
                                const double ca = 1 / std::sqrt(1 + t * t),
                                             cg = std::sqrt(1 - (1 - ca * ca) / (eta * eta));
                                const double rs = (ca - eta * cg) / (ca + eta * cg),
                                             rp = (eta * ca - cg) / (eta * ca + cg);
                                const double F = .5 * (rs * rs + rp * rp);
                                maximumFresnel = std::max(maximumFresnel, F);
                                const double common = weights[ix] * weights[iz] * .0625 * ca * geometric * (1 - F) *
                                                      (1 - F) * std::pow(F, 2 * pairs) * scale;
                                for (std::size_t channel = 0; channel < 3; ++channel)
                                    prediction[channel] += common * std::exp(std::log(double(colors[channel])) *
                                                                             unfoldedDepth / (cg * absorptionDepth()));
                            }
                        }
                    for (auto value : prediction)
                        minimumPrediction = std::min(minimumPrediction, value);
                    absorptionPredictions.push_back({std::size_t{y} * kExtent.width + x, prediction});
                }
            EXPECT_LE(maximumResidual, 1e-12);
            EXPECT_LT(maximumFresnel, .1);
            // For this ROI F<=.1 (and falls toward normal incidence as pairs grow),
            // cos<=1, Beer<=1, G<=1/a^2. The omitted k>=4 reflection-pair tail is
            // bounded geometrically, not silently treated as an exact finite sum.
            const double omittedBound = scale * .25 / (a * a) * std::pow(.1, 8) / (1 - .1 * .1);
            EXPECT_LT(omittedBound / minimumPrediction, 1e-6);
        }
        AbsorptionEnergy result;
        for (const auto& reference : absorptionPredictions) {
            const auto pixel = pixels[reference.pixel];
            const std::array<double, 3> observed{pixel.x, pixel.y, pixel.z};
            for (std::size_t channel = 0; channel < 3; ++channel) {
                EXPECT_TRUE(std::isfinite(observed[channel]));
                result.observed[channel] += observed[channel];
                result.expected[channel] += reference.value[channel];
                result.mse += std::pow(observed[channel] - reference.value[channel], 2);
            }
            ++result.pixels;
        }
        EXPECT_GT(result.pixels, 500U);
        return result;
    }

    Energy sdsEnergy(std::span<const sm::float4> pixels) {
        Energy result;
        const double tangent = std::tan(43 * std::numbers::pi / 360);
        constexpr int subdivisions = 24;
        for (std::uint32_t y = 0; y < kExtent.height; ++y)
            for (std::uint32_t x = 0; x < kExtent.width; ++x) {
                const double u = (2 * (x + .5) / kExtent.width - 1) * tangent * kExtent.width / kExtent.height;
                const double v = (1 - 2 * (y + .5) / kExtent.height) * tangent;
                const double dy = -.6 + .8 * v, dz = .8 + .6 * v, t = -3 / dy;
                const double px = 2 + u * t, pz = 2 + dz * t;
                if (!(t > 0 && px > .6 && px < 2.7 && std::abs(pz - 6) < 1.5))
                    continue;
                double direct = 0, caustic = 0;
                for (int iy = 0; iy < subdivisions; ++iy)
                    for (int iz = 0; iz < subdivisions; ++iz) {
                        const double ly = 1.75 + (iy + .5) * .5 / subdivisions,
                                     lz = 5.75 + (iz + .5) * .5 / subdivisions;
                        const double weight = .25 / (subdivisions * subdivisions);
                        const double rDirect = (3 - px) * (3 - px) + ly * ly + (lz - pz) * (lz - pz);
                        const double rX = (-3 - px) * (-3 - px) + ly * ly + (lz - pz) * (lz - pz);
                        const double rZ = (3 - px) * (3 - px) + ly * ly + (8 - lz - pz) * (8 - lz - pz);
                        direct += weight * ly * (3 - px) / (rDirect * rDirect);
                        caustic += weight * ly * ((3 + px) / (rX * rX) + .4 * (3 - px) / (rZ * rZ));
                    }
                constexpr double scale = .4 * .5 / std::numbers::pi * 10 / 1.2;
                direct *= scale;
                caustic *= scale;
                const auto pixel = pixels[std::size_t{y} * kExtent.width + x];
                const double observed = (double(pixel.x) + pixel.y + pixel.z) / 3;
                EXPECT_TRUE(std::isfinite(observed));
                result.direct += direct;
                result.reflected += caustic;
                result.observed += observed;
                result.mse += std::pow(observed - direct - caustic, 2);
                ++result.pixels;
            }
        EXPECT_GT(result.pixels, 500U);
        return result;
    }

    void compareSlabPrefix(std::uint32_t receiverBudget) {
        const bool bulk = absorptionDepth() > 0;
        const std::uint32_t frames = receiverBudget == 3 ? (bulk ? 4096U : 64U) : 512U;
        std::array<double, 3> observed{}, expected{};
        for (auto seed : {17U, 91U, 1234U, 2026U}) {
            tracer.setConfig(hyperion::PathTracer::Config{.samplesPerPixel = 1,
                                                          .maxDepth = receiverBudget + 1,
                                                          .rngSeed = seed,
                                                          .smsPartitionResolution = 1,
                                                          .smsPriorEnabled = false});
            std::uint64_t continuations = 0;
            for (std::uint32_t f = 0; f < frames; ++f)
                ASSERT_NO_FATAL_FAILURE(frame(f, seed, continuations));
            std::vector<sm::float4> pixels;
            ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
            // Reflection preserves X, while the actual camera's image-X basis is
            // reversed. The centered slab/emitter are symmetric in X. Divide out
            // the independent .4 camera coefficient to use the unchanged oracle.
            for (auto& pixel : pixels)
                pixel = pixel / .4F;
            AbsorptionEnergy measured;
            if (bulk)
                measured = absorptionEnergy(pixels);
            else {
                const auto value = slabEnergy(pixels);
                measured.expected.fill(value.reflected);
                measured.observed.fill(value.observed);
            }
            std::cout << "SMS SDS slab bulk=" << bulk << " budget=" << receiverBudget << " seed=" << seed << " ratios=";
            for (std::size_t channel = 0; channel < 3; ++channel) {
                observed[channel] += measured.observed[channel];
                expected[channel] += measured.expected[channel];
                std::cout << measured.observed[channel] / measured.expected[channel] << ' ';
            }
            std::cout << "continuations=" << continuations << '\n';
        }
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const double target = receiverBudget == 3 && !bulk ? 0.0 : 1.0;
            EXPECT_NEAR(observed[channel] / expected[channel], target, .02)
                << "Receiver-relative TT budget must preserve complementary PT/SMS ownership; channel=" << channel;
        }
    }

    virtual AbsorptionEnergy priorMetric(std::span<const sm::float4> pixels) {
        const auto value = energy(pixels);
        AbsorptionEnergy result;
        result.expected.fill(value.reflected);
        result.observed.fill(value.observed - value.direct);
        result.mse = value.mse;
        result.pixels = value.pixels;
        return result;
    }
    void comparePrior(const char* label) {
        constexpr std::uint32_t frames = 512;
        double plainMse = 0, priorMse = 0;
        std::array<double, 3> expected{}, observed{};
        for (auto seed : {17U, 91U, 1234U, 2026U}) {
            for (bool enabled : {false, true}) {
                tracer.setConfig(hyperion::PathTracer::Config{.samplesPerPixel = 1,
                                                              .maxDepth = glassSlab() ? 4U : 3U,
                                                              .rngSeed = seed,
                                                              .smsPartitionResolution = 16,
                                                              .smsPriorEnabled = enabled});
                std::uint64_t continuations = 0;
                const auto token = seed ^ (enabled ? 0x80000000U : 0U);
                for (std::uint32_t f = 0; f < frames; ++f)
                    ASSERT_NO_FATAL_FAILURE(frame(f, token, continuations));
                std::vector<sm::float4> pixels;
                ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
                const auto measured = priorMetric(pixels);
                std::cout << "SMS prior " << label << " seed=" << seed << " enabled=" << enabled << " ratios=";
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    std::cout << measured.observed[channel] / measured.expected[channel] << ' ';
                    if (enabled) {
                        expected[channel] += measured.expected[channel];
                        observed[channel] += measured.observed[channel];
                    }
                }
                std::cout << "continuations=" << continuations << '\n';
                if (enabled) {
                    priorMse += measured.mse;
                    const auto output = std::filesystem::path(HYPERION_TEST_SHADER_DIR).parent_path() /
                                        (std::string("sms_prior_") + label + "_seed" + std::to_string(seed) + ".exr");
                    ASSERT_TRUE(harmonia::presentation::ImageCapture::saveExr(deviceCtx(), commandPool(), hdr, output));
                } else
                    plainMse += measured.mse;
            }
        }
        for (std::size_t channel = 0; channel < 3; ++channel)
            EXPECT_NEAR(observed[channel] / expected[channel], 1.0, .02) << "Pooled prior energy channel=" << channel;
        std::cout << "SMS prior " << label << " pooled MSE ratio=" << priorMse / plainMse << '\n';
        EXPECT_LT(priorMse, plainMse) << "Pilot prior must improve same-budget image error before claiming benefit";
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
    std::vector<std::pair<std::size_t, double>> slabPredictions;
    std::vector<AbsorptionPrediction> absorptionPredictions;
};

class SmsGlassFrameFixture : public SmsFrameFixture {
  protected:
    [[nodiscard]] bool glassSlab() const noexcept override { return true; }
};

class SmsAbsorptionFrameFixture : public SmsGlassFrameFixture {
  protected:
    [[nodiscard]] float absorptionDepth() const noexcept override { return 1; }
};

class SmsPartitionFrameFixture : public SmsFrameFixture {
  protected:
    [[nodiscard]] std::uint32_t partitionResolution() const noexcept override { return 16; }
};

class SmsSdsFrameFixture : public SmsFrameFixture {
  protected:
    [[nodiscard]] bool specularPrefix() const noexcept override { return true; }
};

class SmsSdsGlassFrameFixture : public SmsGlassFrameFixture {
  protected:
    [[nodiscard]] bool slabPrefix() const noexcept override { return true; }
};
class SmsSdsAbsorptionFrameFixture : public SmsSdsGlassFrameFixture {
  protected:
    [[nodiscard]] float absorptionDepth() const noexcept override { return 1; }
};

class SmsPriorFrameFixture : public SmsPartitionFrameFixture {
  protected:
    [[nodiscard]] bool priorEnabled() const noexcept override { return true; }
};
class SmsPriorGlassFrameFixture : public SmsGlassFrameFixture {
  protected:
    [[nodiscard]] std::uint32_t partitionResolution() const noexcept override { return 16; }
    [[nodiscard]] bool priorEnabled() const noexcept override { return true; }
    AbsorptionEnergy priorMetric(std::span<const sm::float4> pixels) override {
        const auto value = slabEnergy(pixels);
        AbsorptionEnergy result;
        result.expected.fill(value.reflected);
        result.observed.fill(value.observed);
        result.mse = value.mse;
        result.pixels = value.pixels;
        return result;
    }
};
class SmsPriorAbsorptionFrameFixture : public SmsAbsorptionFrameFixture {
  protected:
    [[nodiscard]] std::uint32_t partitionResolution() const noexcept override { return 16; }
    [[nodiscard]] bool priorEnabled() const noexcept override { return true; }
    AbsorptionEnergy priorMetric(std::span<const sm::float4> pixels) override { return absorptionEnergy(pixels); }
};

TEST_F(SmsFrameFixture, MirrorMatchesVirtualEmitterEnergyAcrossSeedsAndConverges) {
    double shortMse = 0, longMse = 0;
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        hyperion::PathTracer::Config config{.samplesPerPixel = 1,
                                            .maxDepth = 3,
                                            .rngSeed = seed,
                                            .smsPartitionResolution = partitionResolution(),
                                            .smsPriorEnabled = priorEnabled()};
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

TEST_F(SmsGlassFrameFixture, ClosedGlassMatchesSnellAreaEnergyAcrossSeedsAndConverges) {
    double shortMse = 0, longMse = 0;
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        tracer.setConfig(hyperion::PathTracer::Config{.samplesPerPixel = 1,
                                                      .maxDepth = 4,
                                                      .rngSeed = seed,
                                                      .smsPartitionResolution = partitionResolution(),
                                                      .smsPriorEnabled = priorEnabled()});
        std::uint64_t continuations = 0;
        std::vector<sm::float4> pixels;
        for (std::uint32_t f = 0; f < kFrames; ++f) {
            ASSERT_NO_FATAL_FAILURE(frame(f, seed, continuations));
            if (f == 31) {
                ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
                shortMse += slabEnergy(pixels).mse;
            }
        }
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = slabEnergy(pixels);
        const double ratio = measured.observed / measured.reflected;
        std::cout << "SMS closed glass seed=" << seed << " transmitted energy ratio=" << ratio
                  << " continuation dispatches=" << continuations << '\n';
        EXPECT_NEAR(ratio, 1.0, .02);
        EXPECT_GT(continuations, kFrames);
        longMse += measured.mse;
        const auto output = std::filesystem::path(HYPERION_TEST_SHADER_DIR).parent_path() /
                            ("sms_closed_glass_seed" + std::to_string(seed) + ".exr");
        ASSERT_TRUE(harmonia::presentation::ImageCapture::saveExr(deviceCtx(), commandPool(), hdr, output));
    }
    std::cout << "SMS closed glass aggregate MSE 128/32=" << longMse / shortMse << '\n';
    EXPECT_LT(longMse, shortMse * .65);
}

TEST_F(SmsAbsorptionFrameFixture, PureAbsorberMatchesSnellBeerRgbEnergyAndConverges) {
    double shortMse = 0, longMse = 0;
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        tracer.setConfig(hyperion::PathTracer::Config{.samplesPerPixel = 1,
                                                      .maxDepth = 4,
                                                      .rngSeed = seed,
                                                      .smsPartitionResolution = partitionResolution(),
                                                      .smsPriorEnabled = priorEnabled()});
        std::uint64_t continuations = 0;
        std::vector<sm::float4> pixels;
        for (std::uint32_t f = 0; f < kFrames; ++f) {
            ASSERT_NO_FATAL_FAILURE(frame(f, seed, continuations));
            if (f == 31) {
                ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
                shortMse += absorptionEnergy(pixels).mse;
            }
        }
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = absorptionEnergy(pixels);
        std::cout << "SMS absorption seed=" << seed << " RGB ratios=";
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const double ratio = measured.observed[channel] / measured.expected[channel];
            std::cout << ratio << ' ';
            EXPECT_NEAR(ratio, 1.0, .02) << "channel=" << channel;
        }
        std::cout << "continuations=" << continuations << '\n';
        EXPECT_GT(continuations, kFrames);
        longMse += measured.mse;
        const auto output = std::filesystem::path(HYPERION_TEST_SHADER_DIR).parent_path() /
                            ("sms_absorption_seed" + std::to_string(seed) + ".exr");
        ASSERT_TRUE(harmonia::presentation::ImageCapture::saveExr(deviceCtx(), commandPool(), hdr, output));
    }
    std::cout << "SMS absorption aggregate MSE 128/32=" << longMse / shortMse << '\n';
    EXPECT_LT(longMse, shortMse * .65);
}

TEST_F(SmsPartitionFrameFixture, PartitionedMirrorMatchesIndependentEnergy) {
    // Hong Section 3.4: without the following prior stage, smaller trust regions
    // trade variance for cost. Use the predeclared larger budget, not the base
    // estimator's 128-sample variance gate or a relaxed energy tolerance.
    constexpr std::uint32_t frames = 4096;
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        tracer.setConfig(hyperion::PathTracer::Config{.samplesPerPixel = 1,
                                                      .maxDepth = 3,
                                                      .rngSeed = seed,
                                                      .smsPartitionResolution = 16,
                                                      .smsPriorEnabled = false});
        std::uint64_t continuations = 0;
        for (std::uint32_t f = 0; f < frames; ++f)
            ASSERT_NO_FATAL_FAILURE(frame(f, seed, continuations));
        std::vector<sm::float4> pixels;
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = energy(pixels);
        const double ratio = (measured.observed - measured.direct) / measured.reflected;
        std::cout << "Partitioned SMS N=16 seed=" << seed << " caustic ratio=" << ratio
                  << " continuations=" << continuations << '\n';
        EXPECT_NEAR(ratio, 1.0, .02);
        EXPECT_GT(continuations, frames);
        const auto output = std::filesystem::path(HYPERION_TEST_SHADER_DIR).parent_path() /
                            ("sms_partition_seed" + std::to_string(seed) + ".exr");
        ASSERT_TRUE(harmonia::presentation::ImageCapture::saveExr(deviceCtx(), commandPool(), hdr, output));
    }
}

TEST_F(SmsPriorFrameFixture, PriorPreservesMirrorEnergyAndImprovesSameBudgetError) {
    ASSERT_NO_FATAL_FAILURE(comparePrior("mirror"));
}
TEST_F(SmsPriorGlassFrameFixture, PriorPreservesGlassEnergyAndImprovesSameBudgetError) {
    ASSERT_NO_FATAL_FAILURE(comparePrior("glass"));
}
TEST_F(SmsPriorAbsorptionFrameFixture, PriorPreservesAbsorptionEnergyAndImprovesSameBudgetError) {
    ASSERT_NO_FATAL_FAILURE(comparePrior("absorption"));
}

TEST_F(SmsSdsFrameFixture, SpecularCameraPrefixUsesCorrectThroughputAndRemainingDepth) {
    for (auto seed : {17U, 91U, 1234U, 2026U}) {
        tracer.setConfig(hyperion::PathTracer::Config{.samplesPerPixel = 1,
                                                      .maxDepth = 4,
                                                      .rngSeed = seed,
                                                      .smsPartitionResolution = 1,
                                                      .smsPriorEnabled = false});
        std::uint64_t continuations = 0;
        for (std::uint32_t f = 0; f < 512; ++f)
            ASSERT_NO_FATAL_FAILURE(frame(f, seed, continuations));
        std::vector<sm::float4> pixels;
        ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
        const auto measured = sdsEnergy(pixels);
        const double ratio = (measured.observed - measured.direct) / measured.reflected;
        std::cout << "SMS SDS seed=" << seed << " caustic ratio=" << ratio << " continuations=" << continuations
                  << '\n';
        EXPECT_NEAR(ratio, 1.0, .02);
        EXPECT_GT(continuations, 512U);
        const auto output = std::filesystem::path(HYPERION_TEST_SHADER_DIR).parent_path() /
                            ("sms_sds_seed" + std::to_string(seed) + ".exr");
        ASSERT_TRUE(harmonia::presentation::ImageCapture::saveExr(deviceCtx(), commandPool(), hdr, output));
    }
}

TEST_F(SmsSdsFrameFixture, InsufficientRemainingDepthRetainsOnlyDirectLighting) {
    tracer.setConfig(hyperion::PathTracer::Config{
        .samplesPerPixel = 1, .maxDepth = 3, .rngSeed = 17, .smsPartitionResolution = 1, .smsPriorEnabled = false});
    std::uint64_t continuations = 0;
    for (std::uint32_t f = 0; f < 64; ++f)
        ASSERT_NO_FATAL_FAILURE(frame(f, 17, continuations));
    std::vector<sm::float4> pixels;
    ASSERT_NO_FATAL_FAILURE(readPixels(pixels));
    const auto measured = sdsEnergy(pixels);
    EXPECT_NEAR((measured.observed - measured.direct) / measured.reflected, 0.0, .02)
        << "A depth-1 receiver has no room for R+L under a three-surface budget";
}

TEST_F(SmsSdsGlassFrameFixture, ThreeReceiverSurfacesCannotContainTTAndEmitter) {
    ASSERT_NO_FATAL_FAILURE(compareSlabPrefix(3));
}
TEST_F(SmsSdsGlassFrameFixture, FourReceiverSurfacesPreserveTransmittedEnergy) {
    ASSERT_NO_FATAL_FAILURE(compareSlabPrefix(4));
}
TEST_F(SmsSdsAbsorptionFrameFixture, ThreeReceiverSurfacesRetainPtOwnedBulkExit) {
    ASSERT_NO_FATAL_FAILURE(compareSlabPrefix(3));
}
TEST_F(SmsSdsAbsorptionFrameFixture, FourReceiverSurfacesPreserveBeerEnergy) {
    ASSERT_NO_FATAL_FAILURE(compareSlabPrefix(4));
}
