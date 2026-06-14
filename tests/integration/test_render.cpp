#include <volk/volk.h>

#include <SDL3/SDL.h>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>

#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "harmonia/utils/Math.hpp"
#include "harmonia/vulkan_init/Context.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

namespace {
class SdlVideoScope {
  public:
    SdlVideoScope() noexcept : m_initialized(SDL_Init(SDL_INIT_VIDEO)) {}

    ~SdlVideoScope() {
        if (m_initialized) {
            SDL_Quit();
        }
    }

    [[nodiscard]] bool initialized() const noexcept { return m_initialized; }

  private:
    bool m_initialized = false;
};

struct WindowDeleter {
    void operator()(SDL_Window* window) const noexcept {
        if (window != nullptr) {
            SDL_DestroyWindow(window);
        }
    }
};

[[nodiscard]] std::filesystem::path shaderRoot() {
#ifdef HYPERION_SHADER_DIR
    return std::filesystem::path(HYPERION_SHADER_DIR);
#else
    return std::filesystem::path("build") / "shaders";
#endif
}

[[nodiscard]] Pipeline::ShaderPaths makeShaderPaths(const std::filesystem::path& root) {
    return Pipeline::ShaderPaths{
        .raygen = root / "raygen.spv",
        .closesthitTriangle = root / "closesthit.spv",
        .closesthitSphere = root / "closesthit.spv",
        .intersection = root / "intersection.spv",
        .miss = root / "miss.spv",
        .shadowMiss = root / "shadow_miss.spv",
    };
}

[[nodiscard]] bool shadersExist(const Pipeline::ShaderPaths& paths) {
    return std::filesystem::exists(paths.raygen) && std::filesystem::exists(paths.closesthitTriangle) &&
           std::filesystem::exists(paths.closesthitSphere) && std::filesystem::exists(paths.intersection) &&
           std::filesystem::exists(paths.miss) && std::filesystem::exists(paths.shadowMiss);
}
} // namespace

TEST(PathTracer, CornellBoxNonBlack) {
    if (const VkResult volkResult = volkInitialize(); volkResult != VK_SUCCESS) {
        GTEST_SKIP() << "Vulkan loader unavailable: VkResult=" << static_cast<int>(volkResult);
    }

    SdlVideoScope sdl;
    if (!sdl.initialized()) {
        GTEST_SKIP() << "SDL video initialization failed: " << SDL_GetError();
    }

    std::unique_ptr<SDL_Window, WindowDeleter> window(
        SDL_CreateWindow("Hyperion Render Test", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN));
    if (!window) {
        GTEST_SKIP() << "Failed to create Vulkan test window: " << SDL_GetError();
    }

    Context::Config config{};
    config.appName = "HyperionTestRender";
    config.enableValidation = false;
    config.window = window.get();

    auto context = Context::create(config);
    if (!context) {
        GTEST_SKIP() << "No Vulkan RT-capable device/context available: VkResult=" << static_cast<int>(context.error());
    }

    const auto shaderPaths = makeShaderPaths(shaderRoot());
    if (!shadersExist(shaderPaths)) {
        GTEST_SKIP() << "Compiled shaders not found under " << shaderRoot().string();
    }

    auto commandPool = CommandPool::create(context->deviceContext(), context->deviceContext().graphicsFamily);
    if (!commandPool) {
        GTEST_SKIP() << "Failed to create command pool: VkResult=" << static_cast<int>(commandPool.error());
    }

    auto descriptors = Descriptors::create(context->deviceContext());
    if (!descriptors) {
        GTEST_SKIP() << "Failed to create descriptors: VkResult=" << static_cast<int>(descriptors.error());
    }

    auto pipeline = Pipeline::create(context->deviceContext(), *descriptors, shaderPaths, 2U);
    if (!pipeline) {
        GTEST_SKIP() << "Failed to create ray tracing pipeline: VkResult=" << static_cast<int>(pipeline.error());
    }

    auto sbt = ShaderBindingTable::create(context->deviceContext(), *pipeline, context->physicalDeviceInfo().rtProps);
    if (!sbt) {
        GTEST_SKIP() << "Failed to create SBT: VkResult=" << static_cast<int>(sbt.error());
    }

    constexpr VkExtent2D renderExtent{64U, 64U};
    auto hdrImage = Image::create(context->deviceContext(),
                                  renderExtent,
                                  VK_FORMAT_R32G32B32A32_SFLOAT,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                  "hyperion.test.hdr");
    if (!hdrImage) {
        GTEST_SKIP() << "Failed to create HDR image: VkResult=" << static_cast<int>(hdrImage.error());
    }

    auto gNormal = Image::create(context->deviceContext(),
                                 renderExtent,
                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_IMAGE_USAGE_STORAGE_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                 "hyperion.test.gNormal");
    if (!gNormal) {
        GTEST_SKIP() << "Failed to create G-buffer normal: VkResult=" << static_cast<int>(gNormal.error());
    }

    auto gDepth = Image::create(context->deviceContext(),
                                renderExtent,
                                VK_FORMAT_R32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                "hyperion.test.gDepth");
    if (!gDepth) {
        GTEST_SKIP() << "Failed to create G-buffer depth: VkResult=" << static_cast<int>(gDepth.error());
    }

    const VkDeviceSize readbackSize =
        static_cast<VkDeviceSize>(renderExtent.width) * renderExtent.height * sizeof(glm::vec4);
    auto readback = Buffer::create(context->deviceContext(),
                                   readbackSize,
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   "hyperion.test.readback");
    if (!readback || readback->mappedData() == nullptr) {
        GTEST_SKIP() << "Failed to create mapped readback buffer";
    }

    Scene scene;
    const uint32_t floorMaterial = scene.addMaterial(Material::diffuse(glm::vec3(0.8F), 1.0F));
    const uint32_t sphereMaterial = scene.addMaterial(Material::metal(glm::vec3(0.9F, 0.3F, 0.2F), 0.15F));
    // Emissive sphere as area light: replaces the removed procedural sky.
    // 50 000 cd/m² at EV100=0 (exposure≈0.833) → display-space ≈ 41 667, clamped to 100
    // by kMaxDisplayLuminance.  At r=1.5, d=6.5 → P(hit)≈1.7 % → average luminance ≫ 1e-3.
    const uint32_t lightMaterial = scene.addMaterial(Material::emissive(glm::vec3(1.0F), 50000.0F));

    const glm::mat4 floorTransform = glm::translate(glm::mat4(1.0F), glm::vec3(0.0F, -1.5F, 0.0F));
    MeshData floorMesh = ProceduralGeometry::makeBox(glm::vec3(4.0F, 0.1F, 4.0F), floorTransform);
    const uint32_t floorInstance =
        scene.addMesh(context->deviceContext(), *commandPool, std::move(floorMesh), floorMaterial, "test.floor");
    if (floorInstance == std::numeric_limits<uint32_t>::max()) {
        GTEST_SKIP() << "Failed to upload floor mesh";
    }

    const uint32_t sphereInstance =
        scene.addSphere(context->deviceContext(), *commandPool, glm::vec3(0.0F, -0.25F, 0.5F), 1.0F, sphereMaterial);
    if (sphereInstance == std::numeric_limits<uint32_t>::max()) {
        GTEST_SKIP() << "Failed to upload sphere";
    }

    // Emissive sphere positioned above scene, fully visible from camera.
    const uint32_t lightInstance =
        scene.addSphere(context->deviceContext(), *commandPool, glm::vec3(0.0F, 5.0F, 0.0F), 1.5F, lightMaterial);
    if (lightInstance == std::numeric_limits<uint32_t>::max()) {
        GTEST_SKIP() << "Failed to upload light sphere";
    }

    if (const VkResult buildResult = scene.build(context->deviceContext(), *commandPool); buildResult != VK_SUCCESS) {
        GTEST_SKIP() << "Failed to build scene acceleration structures: VkResult=" << static_cast<int>(buildResult);
    }

    if (const VkResult descriptorResult = descriptors->updateSceneSet(context->deviceContext(),
                                                                      scene.instanceBuffer().handle(),
                                                                      scene.materialBuffer().handle(),
                                                                      scene.vertexBuffer().handle(),
                                                                      scene.indexBuffer().handle(),
                                                                      scene.lightBuffer().handle(),
                                                                      scene.emissiveTriangleBuffer().handle(),
                                                                      scene.textures());
        descriptorResult != VK_SUCCESS) {
        GTEST_SKIP() << "Failed to update scene descriptor set: VkResult=" << static_cast<int>(descriptorResult);
    }

    auto pathTracer = PathTracer::create(context->deviceContext(),
                                         renderExtent,
                                         *pipeline,
                                         *sbt,
                                         *descriptors,
                                         PathTracer::Config{.samplesPerPixel = 1U, .maxDepth = 2U});
    if (!pathTracer) {
        GTEST_SKIP() << "Failed to create path tracer: VkResult=" << static_cast<int>(pathTracer.error());
    }

    Camera camera(Camera::Params{
        .position = glm::vec3(0.0F, 1.0F, -6.0F),
        .target = glm::vec3(0.0F, -0.1F, 0.2F),
        .up = glm::vec3(0.0F, 1.0F, 0.0F),
        .vfovDeg = 45.0F,
        .aspectRatio = 1.0F,
        .nearPlane = 0.1F,
        .farPlane = 100.0F,
        .lensRadius = 0.0F,
        .focusDist = 6.0F,
        // EV100=0 → exposure=1/1.2≈0.833; correct for 50 000 cd/m² emissive light source.
        .physical = Camera::PhysicalCamera{.aperture = 1.0F, .shutterSpeedHz = 1.0F, .iso = 100.0F},
    });

    auto cmd = commandPool->beginOneShot();
    if (!cmd) {
        GTEST_SKIP() << "Failed to allocate command buffer: VkResult=" << static_cast<int>(cmd.error());
    }

    hdrImage->transition(*cmd,
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         0,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    gNormal->transition(*cmd,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                        0,
                        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    gDepth->transition(*cmd,
                       VK_IMAGE_LAYOUT_UNDEFINED,
                       VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                       0,
                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    ASSERT_EQ(pathTracer->render(*cmd, scene, camera, *hdrImage, *gNormal, *gDepth, 0U), VK_SUCCESS);

    hdrImage->transition(*cmd,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT);

    const VkBufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            VkImageSubresourceLayers{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = VkOffset3D{0, 0, 0},
        .imageExtent = VkExtent3D{renderExtent.width, renderExtent.height, 1U},
    };
    vkCmdCopyImageToBuffer(
        *cmd, hdrImage->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback->handle(), 1U, &copyRegion);

    ASSERT_EQ(commandPool->endOneShot(*cmd), VK_SUCCESS);

    const auto* pixels = static_cast<const glm::vec4*>(readback->mappedData());
    ASSERT_NE(pixels, nullptr);

    double averageLuminance = 0.0;
    const size_t pixelCount = static_cast<size_t>(renderExtent.width) * renderExtent.height;
    for (size_t i = 0; i < pixelCount; ++i) {
        averageLuminance += Math::luminance(glm::max(glm::vec3(pixels[i]), glm::vec3(0.0F)));
    }
    averageLuminance /= static_cast<double>(pixelCount);

    EXPECT_GT(static_cast<float>(averageLuminance), 1.0e-3F);
}
