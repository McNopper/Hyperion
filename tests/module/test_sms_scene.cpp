// Actual production scene buffers + real BLAS/TLAS; independent planar/ellipsoid
// position and finite-difference normal oracles. No CPU copy of the Gram inverse.
#include <volk/volk.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/ShaderModule.hpp"
#include "harmonia/core/VulkanHandle.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "hyperion/ShaderPaths.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

namespace {
struct ReaderInput {
    sm::float4 origin, direction;
};
struct ReaderOutput {
    sm::float4 position, normal, geometric, dpdu, dpdv, dndu, dndv, keys;
};
struct ReaderPush {
    VkDeviceAddress inputs, outputs;
    std::uint32_t count, pad;
};
static_assert(sizeof(ReaderInput) == 32 && sizeof(ReaderOutput) == 128 && sizeof(ReaderPush) == 24);

sm::float3 xyz(sm::float4 value) {
    return {value.x, value.y, value.z};
}
void expectVector(sm::float3 actual, sm::float3 expected, float tolerance = 3.0e-4F) {
    EXPECT_LE(sm::length(actual - expected), tolerance)
        << "actual " << actual.x << ',' << actual.y << ',' << actual.z << " expected " << expected.x << ','
        << expected.y << ',' << expected.z;
}
} // namespace

class SmsSceneFixture : public RtFixture {};

TEST_F(SmsSceneFixture, ProductionReaderMatchesTransformedTriangleAndEllipsoidDifferentials) {
    const auto& ctx = deviceCtx();
    hyperion::Scene scene;
    const auto material = scene.addMaterial(harmonia::scene::Material::mirror());
    harmonia::scene::MeshData data;
    const std::array positions{sm::float3(-1, -1, 0), sm::float3(1, -1, 0), sm::float3(0, 1, 0)};
    const std::array normals{
        sm::float3(0, 0, 1), sm::normalize(sm::float3(.2F, 0, 1)), sm::normalize(sm::float3(0, .3F, 1))};
    for (std::size_t i = 0; i < positions.size(); ++i) {
        harmonia::GpuVertex vertex{};
        vertex.position = positions[i];
        vertex.normal = normals[i];
        data.vertices.push_back(vertex);
    }
    data.indices = {0, 1, 2};
    const auto mesh = scene.addMesh(ctx, commandPool(), std::move(data), {}, "test.sms.triangle");
    ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
    ASSERT_NE(scene.addInstance(
                  mesh, harmonia::scene::Transform{.translation = {-3, 0, 0}, .scale = {1.5F, .75F, 1}}, material),
              std::numeric_limits<std::uint32_t>::max());
    const auto sphere = scene.addSphereMesh(ctx, commandPool(), 1, "test.sms.ellipsoid");
    ASSERT_NE(sphere, std::numeric_limits<std::uint32_t>::max());
    ASSERT_NE(
        scene.addInstance(sphere, harmonia::scene::Transform{.translation = {3, 0, 0}, .scale = {2, 1, .5F}}, material),
        std::numeric_limits<std::uint32_t>::max());
    for (std::uint32_t variant = 0; variant < 4; ++variant) {
        harmonia::scene::MeshData fallback;
        for (std::size_t i = 0; i < positions.size(); ++i) {
            harmonia::GpuVertex vertex{};
            vertex.position = positions[i];
            if (variant == 1 || variant == 2)
                vertex.normal =
                    i == 2 ? sm::normalize(sm::float3(-1, 0, variant == 2 ? 1e-5F : 0)) : sm::float3(1, 0, 0);
            if (variant == 3 && i != 0)
                vertex.normal = sm::normalize(sm::float3(.2F, 0, 1));
            fallback.vertices.push_back(vertex);
        }
        fallback.indices = {0, 1, 2};
        const auto meshId = scene.addMesh(ctx, commandPool(), std::move(fallback), {}, "test.sms.fallback");
        ASSERT_NE(meshId, std::numeric_limits<std::uint32_t>::max());
        ASSERT_NE(scene.addInstance(
                      meshId, harmonia::scene::Transform{.translation = {-9 - 4 * float(variant), 0, 0}}, material),
                  std::numeric_limits<std::uint32_t>::max());
    }
    ASSERT_EQ(scene.build(ctx, commandPool()), VK_SUCCESS);

    auto descriptors = harmonia::renderer::Descriptors::create(ctx);
    ASSERT_TRUE(descriptors.has_value());
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
    auto camera =
        harmonia::core::Buffer::create(ctx,
                                       sizeof(harmonia::CameraData),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       "test.sms.unused.camera");
    ASSERT_TRUE(camera.has_value());
    ASSERT_EQ(descriptors->updateFrameSet(ctx, scene.tlas(), VK_NULL_HANDLE, *camera, VK_NULL_HANDLE, VK_NULL_HANDLE),
              VK_SUCCESS);
    const std::array inputs{ReaderInput{{-3, 0, 3, 0}, {0, 0, -1, 2}},
                            ReaderInput{{-3, 0, -3, 0}, {0, 0, 1, 0}},
                            ReaderInput{{3.5F, .25F, 3, 0}, {0, 0, -1, 0}},
                            ReaderInput{{3, 0, 0, 0}, {0, 0, 1, 0}},
                            ReaderInput{{0, 0, 3, 0}, {0, 0, 1, 0}},
                            ReaderInput{{-9, 0, 3, 0}, {0, 0, -1, 5}},
                            ReaderInput{{-13, 0, 3, 0}, {0, 0, -1, 5}},
                            ReaderInput{{-17, 0, 3, 0}, {0, 0, -1, 5}},
                            ReaderInput{{-21, 0, 3, 0}, {0, 0, -1, 5}}};
    constexpr auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto source = harmonia::core::Buffer::create(
        ctx, sizeof(inputs), storage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.sms.reader.input");
    auto target = harmonia::core::Buffer::create(
        ctx, sizeof(ReaderOutput) * inputs.size(), storage, VMA_MEMORY_USAGE_GPU_TO_CPU, "test.sms.reader.output");
    ASSERT_TRUE(source.has_value() && target.has_value());
    source->uploadData(inputs.data(), sizeof(inputs));
    auto shader = harmonia::core::createShaderModule(
        ctx.device, std::filesystem::path(HYPERION_TEST_SHADER_DIR) / "sms_scene_test.spv");
    ASSERT_TRUE(shader.has_value());
    const harmonia::core::UniqueShaderModule module{ctx.device, *shader};
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = descriptors->pipelineLayout();
    VkPipeline raw{};
    ASSERT_EQ(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &raw), VK_SUCCESS);
    const harmonia::core::UniquePipeline pipeline{ctx.device, raw};
    const auto cmd = commandPool().beginOneShot();
    ASSERT_TRUE(cmd.has_value());
    VkMemoryBarrier2 memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    memory.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(*cmd, &dependency);
    descriptors->bindSceneSet(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
    vkCmdBindPipeline(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    const ReaderPush pc{source->deviceAddress(), target->deviceAddress(), static_cast<std::uint32_t>(inputs.size()), 0};
    vkCmdPushConstants(*cmd, descriptors->pipelineLayout(), VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdDispatch(*cmd, 1, 1, 1);
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(*cmd, &dependency);
    ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
    std::array<ReaderOutput, inputs.size()> output{};
    ASSERT_EQ(target->readData(output.data(), sizeof(output)), VK_SUCCESS);
    for (std::size_t i = 0; i < 4; ++i)
        ASSERT_EQ(output[i].position.w, 1.0F) << i;
    EXPECT_EQ(output[4].position.w, 0.0F);
    expectVector(xyz(output[0].position), {-3, 0, 0});
    expectVector(xyz(output[1].position), {-3, 0, 0});
    expectVector(xyz(output[0].geometric), {0, 0, 1});
    expectVector(xyz(output[1].geometric), {0, 0, 1}); // raw winding, even on back hit

    const auto triangleNormal = [&](sm::float3 p) {
        const float b2 = (p.y + .75F) / 1.5F;
        const float b1 = (p.x + 4.5F - 1.5F * b2) / 3.0F;
        const auto inverseScale = sm::float3(1 / 1.5F, 1 / .75F, 1);
        return sm::normalize(sm::normalize(normals[0] * inverseScale) * (1 - b1 - b2) +
                             sm::normalize(normals[1] * inverseScale) * b1 +
                             sm::normalize(normals[2] * inverseScale) * b2);
    };
    constexpr float epsilon = 1.0e-3F;
    const auto checkDifferentials = [&](const ReaderOutput& actual, const auto& normalAt) {
        const auto p = xyz(actual.position), u = xyz(actual.dpdu), v = xyz(actual.dpdv);
        expectVector(xyz(actual.normal), normalAt(p));
        expectVector(xyz(actual.dndu), (normalAt(p + u * epsilon) - normalAt(p - u * epsilon)) / (2 * epsilon));
        expectVector(xyz(actual.dndv), (normalAt(p + v * epsilon) - normalAt(p - v * epsilon)) / (2 * epsilon));
        EXPECT_NEAR(sm::dot(u, xyz(actual.geometric)), 0, 3e-4);
        EXPECT_NEAR(sm::dot(v, xyz(actual.geometric)), 0, 3e-4);
    };
    checkDifferentials(output[0], triangleNormal);
    const float z = .5F * std::sqrt(1 - .25F * .25F - .25F * .25F);
    expectVector(xyz(output[2].position), {3.5F, .25F, z});
    expectVector(xyz(output[3].position), {3, 0, .5F});
    const auto sphereNormal = [](sm::float3 p) {
        const auto d = p - sm::float3(3, 0, 0);
        return sm::normalize(d * sm::float3(.25F, 1, 4));
    };
    checkDifferentials(output[2], sphereNormal);
    checkDifferentials(output[3], sphereNormal);
    EXPECT_EQ(output[0].keys.x, 0.0F);
    EXPECT_EQ(output[2].keys.x, 1.0F);
    EXPECT_EQ(output[2].keys.y, static_cast<float>(material));
    for (std::size_t i = 5; i < inputs.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(output[i].position.w, 1.0F) << "Production normal fallback must not become an SMS miss";
        expectVector(xyz(output[i].normal), {0, 0, 1});
        expectVector(xyz(output[i].dndu), {0, 0, 0});
        expectVector(xyz(output[i].dndv), {0, 0, 0});
    }

    auto visibility = commandPool().beginOneShot();
    ASSERT_TRUE(visibility.has_value());
    memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    memory.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    vkCmdPipelineBarrier2(*visibility, &dependency);
    descriptors->bindSceneSet(*visibility, VK_PIPELINE_BIND_POINT_COMPUTE);
    vkCmdBindPipeline(*visibility, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    const ReaderPush shadow{
        source->deviceAddress(), target->deviceAddress(), static_cast<std::uint32_t>(inputs.size()), 1};
    vkCmdPushConstants(*visibility, descriptors->pipelineLayout(), VK_SHADER_STAGE_ALL, 0, sizeof(shadow), &shadow);
    vkCmdDispatch(*visibility, 1, 1, 1);
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(*visibility, &dependency);
    ASSERT_EQ(commandPool().endOneShot(*visibility), VK_SUCCESS);
    ASSERT_EQ(target->readData(output.data(), sizeof(output)), VK_SUCCESS);
    for (std::size_t i = 0; i < inputs.size(); ++i)
        EXPECT_EQ(output[i].position.w, i == 0 || i == 4 ? 0.0F : 1.0F)
            << "Opaque visibility must not depend on shading normals; ray=" << i;

    // Independent actual-RT control: its primary G-buffer normal must obey the
    // same ellipsoid optics, not the old world-radial sphere approximation.
    auto rt = harmonia::renderer::Pipeline::create(
        ctx, *descriptors, hyperion::makeHyperionShaderPaths(std::filesystem::path(HYPERION_SHADER_DIR)), 2);
    ASSERT_TRUE(rt.has_value());
    auto sbt = hyperion::ShaderBindingTable::create(ctx, *rt, physInfo().rtProps);
    ASSERT_TRUE(sbt.has_value());
    const VkExtent2D one{1, 1};
    auto hdr = harmonia::core::Image::create(ctx,
                                             one,
                                             VK_FORMAT_R32G32B32A32_SFLOAT,
                                             VK_IMAGE_USAGE_STORAGE_BIT,
                                             VK_IMAGE_ASPECT_COLOR_BIT,
                                             "test.sms.rt.hdr");
    auto gn = harmonia::core::Image::create(ctx,
                                            one,
                                            VK_FORMAT_R32G32B32A32_SFLOAT,
                                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                            VK_IMAGE_ASPECT_COLOR_BIT,
                                            "test.sms.rt.normal");
    auto gd = harmonia::core::Image::create(
        ctx, one, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, "test.sms.rt.depth");
    ASSERT_TRUE(hdr.has_value() && gn.has_value() && gd.has_value());
    auto tracer = hyperion::PathTracer::create(
        ctx, one, *rt, *sbt, *descriptors, hyperion::PathTracer::Config{.samplesPerPixel = 1, .maxDepth = 1});
    ASSERT_TRUE(tracer.has_value());
    harmonia::renderer::Camera rtCamera(harmonia::renderer::Camera::Params{.position = {3.5F, .25F, 3},
                                                                           .target = {3.5F, .25F, 0},
                                                                           .up = {0, 1, 0},
                                                                           .vfovDeg = .0001F,
                                                                           .aspectRatio = 1,
                                                                           .nearPlane = .01F,
                                                                           .farPlane = 100});
    auto normalBuffer = harmonia::core::Buffer::create(ctx,
                                                       sizeof(sm::float4),
                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                       VMA_MEMORY_USAGE_GPU_TO_CPU,
                                                       "test.sms.rt.normal.readback");
    ASSERT_TRUE(normalBuffer.has_value());
    auto render = commandPool().beginOneShot();
    ASSERT_TRUE(render.has_value());
    for (auto* image : {&*hdr, &*gn, &*gd})
        image->transition(*render,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_NONE,
                          0,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_SHADER_WRITE_BIT);
    ASSERT_EQ(tracer->render(*render, scene, rtCamera, *hdr, *gn, *gd, 0), VK_SUCCESS);
    gn->transition(*render,
                   VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COPY_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {1, 1, 1};
    vkCmdCopyImageToBuffer(
        *render, gn->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, normalBuffer->handle(), 1, &copy);
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(*render, &dependency);
    ASSERT_EQ(commandPool().endOneShot(*render), VK_SUCCESS);
    const auto ready = tracer->frameReady();
    ASSERT_TRUE(ready.has_value() && *ready);
    sm::float4 rtNormal{};
    ASSERT_EQ(normalBuffer->readData(&rtNormal, sizeof(rtNormal)), VK_SUCCESS);
    expectVector(xyz(rtNormal), sphereNormal({3.5F, .25F, z}));
    for (std::size_t i = 5; i < inputs.size(); ++i) {
        SCOPED_TRACE(i);
        auto camera = harmonia::renderer::Camera::Params{.position = xyz(inputs[i].origin),
                                                         .target = xyz(inputs[i].origin) + sm::float3(0, 0, -3),
                                                         .up = {0, 1, 0},
                                                         .vfovDeg = .0001F,
                                                         .aspectRatio = 1,
                                                         .nearPlane = .01F,
                                                         .farPlane = 100};
        const harmonia::renderer::Camera fallbackCamera(camera);
        auto renderFallback = commandPool().beginOneShot();
        ASSERT_TRUE(renderFallback.has_value());
        gn->transition(*renderFallback,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT);
        ASSERT_EQ(tracer->render(*renderFallback, scene, fallbackCamera, *hdr, *gn, *gd, static_cast<std::uint32_t>(i)),
                  VK_SUCCESS);
        gn->transition(*renderFallback,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT);
        vkCmdCopyImageToBuffer(
            *renderFallback, gn->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, normalBuffer->handle(), 1, &copy);
        memory.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        vkCmdPipelineBarrier2(*renderFallback, &dependency);
        ASSERT_EQ(commandPool().endOneShot(*renderFallback), VK_SUCCESS);
        ASSERT_EQ(normalBuffer->readData(&rtNormal, sizeof(rtNormal)), VK_SUCCESS);
        expectVector(xyz(rtNormal), {0, 0, 1});
    }
}

TEST_F(SmsSceneFixture, ProductionSlabRootsMatchSnellFromOffRootSeeds) {
    struct Input {
        sm::float4 receiver, light, seed;
    };
    struct Output {
        sm::float4 first, second, metrics;
    };
    const auto& ctx = deviceCtx();
    hyperion::Scene scene;
    const auto glass = scene.addMaterial(harmonia::scene::Material::glass(1.5F, 0));
    const auto mesh = scene.addMesh(
        ctx, commandPool(), harmonia::scene::ProceduralGeometry::makeBox({20, 20, .4F}), {}, "test.sms.closed.slab");
    ASSERT_NE(mesh, std::numeric_limits<std::uint32_t>::max());
    ASSERT_NE(scene.addInstance(mesh, harmonia::scene::Transform{.translation = {0, 0, -.4F}}, glass),
              std::numeric_limits<std::uint32_t>::max());
    ASSERT_EQ(scene.build(ctx, commandPool()), VK_SUCCESS);
    auto descriptors = harmonia::renderer::Descriptors::create(ctx);
    ASSERT_TRUE(descriptors.has_value());
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
    auto camera =
        harmonia::core::Buffer::create(ctx,
                                       sizeof(harmonia::CameraData),
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       "test.sms.root.unused.camera");
    ASSERT_TRUE(camera.has_value());
    ASSERT_EQ(descriptors->updateFrameSet(ctx, scene.tlas(), VK_NULL_HANDLE, *camera, VK_NULL_HANDLE, VK_NULL_HANDLE),
              VK_SUCCESS);
    std::vector<Input> inputs;
    std::vector<sm::float3> firstRoots, secondRoots;
    std::vector<double> expectedGeometry;
    constexpr double eta = 1.5, h0 = 2, h2 = 3, thickness = .8;
    for (double degrees : {0, 25, 55}) {
        const auto angle = degrees * std::numbers::pi / 180;
        const auto inside = std::asin(std::sin(angle) / eta);
        const double first = h0 * std::tan(angle), second = first + thickness * std::tan(inside);
        const double radius = second + h2 * std::tan(angle);
        const double derivative =
            (h0 + h2) / std::pow(std::cos(angle), 3) + thickness / eta / std::pow(std::cos(inside), 3);
        const double geometry = degrees == 0 ? 1 / std::pow(h0 + h2 + thickness / eta, 2)
                                             : std::sin(angle) / (std::cos(angle) * radius * derivative);
        for (auto seed : {sm::float3(0, 0, 0), sm::float3(.5F, .2F, 0), sm::float3(-.8F, -.4F, 0)}) {
            inputs.push_back({{0, 0, 2, 0}, {static_cast<float>(radius), 0, -3.8F, 0}, sm::float4(seed, 0)});
            firstRoots.push_back({static_cast<float>(first), 0, 0});
            secondRoots.push_back({static_cast<float>(second), 0, -.8F});
            expectedGeometry.push_back(geometry);
        }
    }
    constexpr auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto source = harmonia::core::Buffer::create(
        ctx, inputs.size() * sizeof(Input), storage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.sms.root.inputs");
    auto target = harmonia::core::Buffer::create(
        ctx, inputs.size() * sizeof(Output), storage, VMA_MEMORY_USAGE_GPU_TO_CPU, "test.sms.root.outputs");
    ASSERT_TRUE(source.has_value() && target.has_value());
    source->uploadData(inputs.data(), inputs.size() * sizeof(Input));
    auto module = harmonia::core::createShaderModule(
        ctx.device, std::filesystem::path(HYPERION_TEST_SHADER_DIR) / "sms_scene_root_test.spv");
    ASSERT_TRUE(module.has_value());
    const harmonia::core::UniqueShaderModule shader{ctx.device, *module};
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = shader;
    info.stage.pName = "main";
    info.layout = descriptors->pipelineLayout();
    VkPipeline raw{};
    ASSERT_EQ(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &raw), VK_SUCCESS);
    const harmonia::core::UniquePipeline pipeline{ctx.device, raw};
    auto cmd = commandPool().beginOneShot();
    ASSERT_TRUE(cmd.has_value());
    VkMemoryBarrier2 memory{};
    memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memory.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    memory.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(*cmd, &dependency);
    descriptors->bindSceneSet(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
    vkCmdBindPipeline(*cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    const ReaderPush pc{source->deviceAddress(), target->deviceAddress(), static_cast<std::uint32_t>(inputs.size()), 0};
    vkCmdPushConstants(*cmd, descriptors->pipelineLayout(), VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
    vkCmdDispatch(*cmd, 1, 1, 1);
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    memory.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(*cmd, &dependency);
    ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);
    std::vector<Output> outputs(inputs.size());
    ASSERT_EQ(target->readData(outputs.data(), outputs.size() * sizeof(Output)), VK_SUCCESS);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        SCOPED_TRACE(i);
        ASSERT_EQ(outputs[i].first.w, 1.0F);
        ASSERT_EQ(outputs[i].second.w, 1.0F);
        expectVector(xyz(outputs[i].first), firstRoots[i]);
        expectVector(xyz(outputs[i].second), secondRoots[i]);
        EXPECT_LE(outputs[i].metrics.y, 1.0e-5F);
        EXPECT_LE(std::abs(outputs[i].metrics.x / expectedGeometry[i] - 1), 2.0e-5);
        EXPECT_EQ(outputs[i].metrics.z, 1.5F);
        EXPECT_EQ(outputs[i].metrics.w, 1 / 1.5F);
    }
}
