#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "demo/Application.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "demo/SceneLoader.hpp"
#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/Logger.hpp"
#include "hyperion/scene/Material.hpp"
#include "hyperion/utils/ColorSpace.hpp"
#include "hyperion/utils/ToneMapping.hpp"

#include <stb_image_write.h>

#ifdef HYPERION_HAS_OPENEXR
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#endif

namespace {
[[nodiscard]] VkResult createBinarySemaphore(VkDevice device, VkSemaphore& semaphore) {
    const VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    return vkCreateSemaphore(device, &info, nullptr, &semaphore);
}

[[nodiscard]] VkResult createTimelineSemaphore(VkDevice device, VkSemaphore& semaphore) {
    const VkSemaphoreTypeCreateInfo typeInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    const VkSemaphoreCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &typeInfo,
        .flags = 0,
    };
    return vkCreateSemaphore(device, &info, nullptr, &semaphore);
}

[[nodiscard]] VkImageMemoryBarrier2 imageBarrier(VkImage image,
                                                 VkImageLayout oldLayout,
                                                 VkImageLayout newLayout,
                                                 VkPipelineStageFlags2 srcStage,
                                                 VkAccessFlags2 srcAccess,
                                                 VkPipelineStageFlags2 dstStage,
                                                 VkAccessFlags2 dstAccess) {
    return VkImageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange =
            VkImageSubresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };
}

void pipelineBarrier(VkCommandBuffer cmd, std::span<const VkImageMemoryBarrier2> imageBarriers) {
    const VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 0,
        .pMemoryBarriers = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = static_cast<uint32_t>(imageBarriers.size()),
        .pImageMemoryBarriers = imageBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &dependencyInfo);
}

[[nodiscard]] std::filesystem::path resolveShaderDir(std::filesystem::path shaderDir) {
    // Explicit user path takes priority only if it actually exists.
    if (!shaderDir.empty() && std::filesystem::exists(shaderDir)) {
        Logger::info("Using shader dir (user-specified): {}", shaderDir.string());
        return shaderDir;
    }
    // Prefer the compile-time output dir (build/shaders) over any source-tree
    // "shaders/" that lacks .spv files.
#ifdef HYPERION_SHADER_DIR
    const std::filesystem::path builtDir = HYPERION_SHADER_DIR;
    if (std::filesystem::exists(builtDir)) {
        Logger::info("Using shader dir (built): {}", builtDir.string());
        return builtDir;
    }
#endif
    Logger::error("No shader directory found (tried '{}' and HYPERION_SHADER_DIR)", shaderDir.string());
    return shaderDir;
}

[[nodiscard]] VkExtent2D windowPixelExtent(SDL_Window* window) {
    int width = 0;
    int height = 0;
    SDL_GetWindowSizeInPixels(window, &width, &height);
    return VkExtent2D{
        .width = static_cast<uint32_t>(std::max(width, 1)),
        .height = static_cast<uint32_t>(std::max(height, 1)),
    };
}
} // namespace

std::expected<std::unique_ptr<Application>, int> Application::create(Config config) {
    auto appPtr = std::make_unique<Application>();
    Application& app = *appPtr;
    app.m_config = std::move(config);

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        Logger::error("SDL_Init failed: {}", SDL_GetError());
        return std::unexpected(1);
    }

    app.m_window = SDL_CreateWindow(app.m_config.title.c_str(),
                                    static_cast<int>(app.m_config.width),
                                    static_cast<int>(app.m_config.height),
                                    SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                                        SDL_WINDOW_HIGH_PIXEL_DENSITY |
                                        (app.m_config.outputFile.empty() ? 0u : SDL_WINDOW_HIDDEN));
    if (app.m_window == nullptr) {
        Logger::error("SDL_CreateWindow failed: {}", SDL_GetError());
        return std::unexpected(1);
    }

    auto context = Context::create(Context::Config{
        .appName = app.m_config.title,
        .appVersion = VK_MAKE_VERSION(1, 0, 0),
        .enableValidation = app.m_config.validation,
        .window = app.m_window,
    });
    if (!context) {
        Logger::error("Context creation failed: VkResult {}", static_cast<int>(context.error()));
        return std::unexpected(1);
    }
    app.m_context = std::move(*context);

    auto pool = CommandPool::create(app.m_context.deviceContext(), app.m_context.physicalDeviceInfo().graphicsFamily);
    if (!pool) {
        Logger::error("CommandPool creation failed: VkResult {}", static_cast<int>(pool.error()));
        return std::unexpected(1);
    }
    app.m_commandPool = std::move(*pool);

    const VkExtent2D initialExtent = windowPixelExtent(app.m_window);
    auto swapchain = Swapchain::create(app.m_context.deviceContext(), app.m_context.surface(), initialExtent, true);
    if (!swapchain) {
        Logger::error("Swapchain creation failed: VkResult {}", static_cast<int>(swapchain.error()));
        return std::unexpected(1);
    }
    app.m_swapchain = std::move(*swapchain);
    app.m_swapchainLayouts.assign(app.m_swapchain.imageCount(), VK_IMAGE_LAYOUT_UNDEFINED);

    auto descriptors = Descriptors::create(app.m_context.deviceContext());
    if (!descriptors) {
        Logger::error("Descriptor creation failed: VkResult {}", static_cast<int>(descriptors.error()));
        return std::unexpected(1);
    }
    app.m_descriptors = std::move(*descriptors);

    const std::filesystem::path shaderDir = resolveShaderDir(app.m_config.shaderDir);
    const Pipeline::ShaderPaths shaderPaths{
        .raygen             = shaderDir / "raygen.spv",
        .closesthitTriangle = shaderDir / "closesthit.spv",
        .closesthitSphere   = shaderDir / "closesthit.spv",
        .intersection       = shaderDir / "intersection.spv",
        .miss               = shaderDir / "miss.spv",
        .shadowMiss         = shaderDir / "shadow_miss.spv",
    };
    auto pipeline =
        Pipeline::create(app.m_context.deviceContext(), app.m_descriptors, shaderPaths, app.m_config.maxDepth);
    if (!pipeline) {
        Logger::error("Pipeline creation failed: VkResult {}", static_cast<int>(pipeline.error()));
        return std::unexpected(1);
    }
    app.m_pipeline = std::move(*pipeline);

    auto sbt = ShaderBindingTable::create(
        app.m_context.deviceContext(), app.m_pipeline, app.m_context.physicalDeviceInfo().rtProps);
    if (!sbt) {
        Logger::error("SBT creation failed: VkResult {}", static_cast<int>(sbt.error()));
        return std::unexpected(1);
    }
    app.m_sbt = std::move(*sbt);

    auto toneMapper = ToneMapper::create(app.m_context.deviceContext(),
                                         app.m_descriptors.pipelineLayout(),
                                         app.m_swapchain.format(),
                                         shaderDir / "tonemap_vert.spv",
                                         shaderDir / "tonemap.spv");
    if (!toneMapper) {
        Logger::error("Tone mapper creation failed: VkResult {}", static_cast<int>(toneMapper.error()));
        return std::unexpected(1);
    }
    app.m_toneMapper = std::move(*toneMapper);

    auto hdrImage = Image::create(app.m_context.deviceContext(),
                                  app.m_swapchain.extent(),
                                  VK_FORMAT_R32G32B32A32_SFLOAT,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                  "demo.hdr");
    if (!hdrImage) {
        Logger::error("HDR image creation failed: VkResult {}", static_cast<int>(hdrImage.error()));
        return std::unexpected(1);
    }
    app.m_hdrImage = std::move(*hdrImage);

    auto gNormal = Image::create(app.m_context.deviceContext(),
                                 app.m_swapchain.extent(),
                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_IMAGE_USAGE_STORAGE_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                 "demo.gNormal");
    if (!gNormal) {
        Logger::error("G-buffer normal creation failed: VkResult {}", static_cast<int>(gNormal.error()));
        return std::unexpected(1);
    }
    app.m_gNormal = std::move(*gNormal);

    auto gDepth = Image::create(app.m_context.deviceContext(),
                                app.m_swapchain.extent(),
                                VK_FORMAT_R32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                "demo.gDepth");
    if (!gDepth) {
        Logger::error("G-buffer depth creation failed: VkResult {}", static_cast<int>(gDepth.error()));
        return std::unexpected(1);
    }
    app.m_gDepth = std::move(*gDepth);

    auto initCmd = app.m_commandPool.beginOneShot();
    if (!initCmd) {
        Logger::error("Init command buffer allocation failed: VkResult {}", static_cast<int>(initCmd.error()));
        return std::unexpected(1);
    }
    app.m_hdrImage.transition(*initCmd,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_NONE,
                              0,
                              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                              VK_ACCESS_2_SHADER_WRITE_BIT);
    app.m_gNormal.transition(*initCmd,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_NONE,
                             0,
                             VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_ACCESS_2_SHADER_WRITE_BIT);
    app.m_gDepth.transition(*initCmd,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_NONE,
                            0,
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                            VK_ACCESS_2_SHADER_WRITE_BIT);
    if (const VkResult result = app.m_commandPool.endOneShot(*initCmd); result != VK_SUCCESS) {
        Logger::error("HDR image transition failed: VkResult {}", static_cast<int>(result));
        return std::unexpected(1);
    }

    // Load scene file — populates geometry and overrides camera / render config.
    SceneLoader loader;
    const auto  sceneConfig = loader.load(
        app.m_config.sceneFile, app.m_config.assetsDir, app.m_scene, app.m_context.deviceContext(), app.m_commandPool);
    if (!sceneConfig) {
        Logger::error("Scene load failed");
        return std::unexpected(1);
    }

    if (sceneConfig->spp && !app.m_config.sppExplicit)
        app.m_config.spp = *sceneConfig->spp;
    if (sceneConfig->maxDepth)
        app.m_config.maxDepth = *sceneConfig->maxDepth;
    const float envLuminance = sceneConfig->envUnitNits.value_or(1.0f);

    // Build camera — fall back to Cornell box defaults when scene file omits settings.
    Camera::PhysicalCamera physical{};
    if (sceneConfig->cameraEv100) {
        // EV100 = log2(N² × t_inv × 100/ISO) with N=1, ISO=100 → t_inv = 2^EV100
        const float ev100 = *sceneConfig->cameraEv100;
        physical.aperture = 1.0f;
        physical.iso = 100.0f;
        physical.shutterSpeedHz = std::pow(2.0f, ev100);
    }
    app.m_camera = Camera(Camera::Params{
        .position    = sceneConfig->cameraPos.value_or(glm::vec3(278.0f, 273.0f, -800.0f)),
        .target      = sceneConfig->cameraAt.value_or(glm::vec3(278.0f, 273.0f, 279.5f)),
        .up          = sceneConfig->cameraUp.value_or(glm::vec3(0.0f, 1.0f, 0.0f)),
        .vfovDeg     = sceneConfig->cameraVfov.value_or(39.1f),
        .aspectRatio = static_cast<float>(app.m_swapchain.extent().width) /
                       static_cast<float>(app.m_swapchain.extent().height),
        .nearPlane  = 0.1f,
        .farPlane   = 10000.0f,
        .lensRadius = 0.0f,
        .focusDist  = 1079.5f,
        .physical   = physical,
    });
    if (const VkResult result = app.m_scene.build(app.m_context.deviceContext(), app.m_commandPool);
        result != VK_SUCCESS) {
        Logger::error("Scene build failed: VkResult {}", static_cast<int>(result));
        return std::unexpected(1);
    }
    Logger::info("Scene built (BLAS+TLAS)");
    if (const VkResult result = app.m_descriptors.updateSceneSet(app.m_context.deviceContext(), app.m_scene);
        result != VK_SUCCESS) {
        Logger::error("Descriptor update failed: VkResult {}", static_cast<int>(result));
        return std::unexpected(1);
    }
    Logger::info("Descriptors updated");

    // Load IBL environment map if specified in the scene file.
    if (sceneConfig->envMapFile) {
        const auto envPath = app.m_config.assetsDir / *sceneConfig->envMapFile;
        auto probe = IblProbe::loadFromEXR(app.m_context.deviceContext(), app.m_commandPool, envPath);
        if (!probe) {
            Logger::warn("IBL probe load failed for '{}' — using procedural sky", envPath.string());
        } else {
            app.m_iblProbe = std::move(*probe);
            if (const VkResult result = app.m_descriptors.updateEnvMap(
                    app.m_context.deviceContext(), app.m_iblProbe.imageView(), app.m_iblProbe.sampler());
                result != VK_SUCCESS) {
                Logger::warn("IBL descriptor update failed: VkResult {}", static_cast<int>(result));
            } else {
                Logger::info("IBL probe loaded: '{}'", envPath.filename().string());
            }
        }
    }

    auto tracer = PathTracer::create(app.m_context.deviceContext(),
                                     app.m_swapchain.extent(),
                                     app.m_pipeline,
                                     app.m_sbt,
                                     app.m_descriptors,
                                     PathTracer::Config{
                                         .samplesPerPixel = app.m_config.spp,
                                         .maxDepth = app.m_config.maxDepth,
                                         .envLuminance = envLuminance,
                                         .hasEnvMap = app.m_iblProbe.isValid() ? 1u : 0u,
                                     });
    if (!tracer) {
        Logger::error("PathTracer creation failed: VkResult {}", static_cast<int>(tracer.error()));
        return std::unexpected(1);
    }
    app.m_pathTracer = std::move(*tracer);
    Logger::info("PathTracer created");

    for (FrameResources& frame : app.m_frames) {
        auto traceCmd = app.m_commandPool.allocate();
        if (!traceCmd) {
            Logger::error("Command buffer allocation failed: VkResult {}", static_cast<int>(traceCmd.error()));
            return std::unexpected(1);
        }
        frame.traceCmd = *traceCmd;

        auto displayCmd = app.m_commandPool.allocate();
        if (!displayCmd) {
            Logger::error("Command buffer allocation failed: VkResult {}", static_cast<int>(displayCmd.error()));
            return std::unexpected(1);
        }
        frame.displayCmd = *displayCmd;

        if (createBinarySemaphore(app.m_context.deviceContext().device, frame.imageAvailable) != VK_SUCCESS) {
            Logger::error("Semaphore creation failed");
            return std::unexpected(1);
        }
    }
    app.m_renderComplete.resize(app.m_swapchain.imageCount());
    for (VkSemaphore& sem : app.m_renderComplete) {
        if (createBinarySemaphore(app.m_context.deviceContext().device, sem) != VK_SUCCESS) {
            Logger::error("renderComplete semaphore creation failed");
            return std::unexpected(1);
        }
    }
    if (createTimelineSemaphore(app.m_context.deviceContext().device, app.m_timelineSemaphore) != VK_SUCCESS) {
        Logger::error("Timeline semaphore creation failed");
        return std::unexpected(1);
    }

    app.m_running = true;
    return appPtr;
}

Application::Application(Application&& other) noexcept
    : m_config(std::move(other.m_config)),
      m_window(std::exchange(other.m_window, nullptr)),
      m_context(std::move(other.m_context)),
      m_commandPool(std::move(other.m_commandPool)),
      m_swapchain(std::move(other.m_swapchain)),
      m_descriptors(std::move(other.m_descriptors)),
      m_pipeline(std::move(other.m_pipeline)),
      m_sbt(std::move(other.m_sbt)),
      m_toneMapper(std::move(other.m_toneMapper)),
      m_pathTracer(std::move(other.m_pathTracer)),
      m_scene(std::move(other.m_scene)),
      m_camera(other.m_camera),
      m_hdrImage(std::move(other.m_hdrImage)),
      m_gNormal(std::move(other.m_gNormal)),
      m_gDepth(std::move(other.m_gDepth)),
      m_iblProbe(std::move(other.m_iblProbe)),
      m_frames(other.m_frames),
      m_renderComplete(std::move(other.m_renderComplete)),
      m_timelineSemaphore(std::exchange(other.m_timelineSemaphore, VK_NULL_HANDLE)),
      m_nextTimelineValue(other.m_nextTimelineValue),
      m_currentFrame(other.m_currentFrame),
      m_frameIndex(other.m_frameIndex),
      m_swapchainLayouts(std::move(other.m_swapchainLayouts)),
      m_running(other.m_running) {
    other.m_frames = {};
    other.m_nextTimelineValue = 1;
    other.m_currentFrame = 0;
    other.m_frameIndex = 0;
    other.m_running = false;
}

Application& Application::operator=(Application&& other) noexcept {
    if (this != &other) {
        destroy();
        m_config = std::move(other.m_config);
        m_window = std::exchange(other.m_window, nullptr);
        m_context = std::move(other.m_context);
        m_commandPool = std::move(other.m_commandPool);
        m_swapchain = std::move(other.m_swapchain);
        m_descriptors = std::move(other.m_descriptors);
        m_pipeline = std::move(other.m_pipeline);
        m_sbt = std::move(other.m_sbt);
        m_toneMapper = std::move(other.m_toneMapper);
        m_pathTracer = std::move(other.m_pathTracer);
        m_scene = std::move(other.m_scene);
        m_camera = other.m_camera;
        m_hdrImage = std::move(other.m_hdrImage);
        m_gNormal = std::move(other.m_gNormal);
        m_gDepth = std::move(other.m_gDepth);
        m_iblProbe = std::move(other.m_iblProbe);
        m_frames = other.m_frames;
        m_renderComplete = std::move(other.m_renderComplete);
        m_timelineSemaphore = std::exchange(other.m_timelineSemaphore, VK_NULL_HANDLE);
        m_nextTimelineValue = other.m_nextTimelineValue;
        m_currentFrame = other.m_currentFrame;
        m_frameIndex = other.m_frameIndex;
        m_swapchainLayouts = std::move(other.m_swapchainLayouts);
        m_running = other.m_running;
        other.m_frames = {};
        other.m_nextTimelineValue = 1;
        other.m_currentFrame = 0;
        other.m_frameIndex = 0;
        other.m_running = false;
    }
    return *this;
}

Application::~Application() {
    destroy();
}

int Application::run() {
    if (!m_config.outputFile.empty()) {
        // Headless: accumulate spp samples, then save.
        Logger::info("Headless render: {} spp -> {}", m_config.spp, m_config.outputFile.string());
        for (uint32_t i = 0; i < m_config.spp; ++i) {
            renderFrame(m_hdrImage);
        }
        vkDeviceWaitIdle(m_context.deviceContext().device);
        saveEXR(m_config.outputFile);
        // Also write a tone-mapped sRGB PNG (ACES SDR) for GitHub / README display.
        auto pngPath = m_config.outputFile;
        pngPath.replace_extension(".png");
        savePNG(pngPath);
        return 0;
    }

    // Interactive: acquire → trace → tonemap → present per frame.
    Logger::info("Interactive render loop (spp={})", m_config.spp);
    while (m_running) {
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                m_running = false;
            } else if (event.type == SDL_EVENT_WINDOW_RESIZED ||
                       event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                handleResize(static_cast<uint32_t>(event.window.data1),
                             static_cast<uint32_t>(event.window.data2));
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_ESCAPE) {
                    m_running = false;
                } else if (event.key.key == SDLK_S) {
                    saveEXR(std::format("hyperion_{:06}.exr", m_frameIndex));
                }
            }
        }

        if (!m_running || m_swapchain.extent().width == 0U || m_swapchain.extent().height == 0U) {
            continue;
        }

        // Save the frame slot before renderFrame advances m_currentFrame.
        const uint32_t slot = m_currentFrame;

        // Submit path trace — signals timeline when the hdr buffer is ready.
        const uint64_t traceValue = renderFrame(m_hdrImage);

        FrameResources& frame = m_frames[slot];

        // Acquire the swapchain image (application concern).
        uint32_t imageIndex = 0;
        VkResult result = m_swapchain.acquireNextImage(frame.imageAvailable, imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            handleResize(m_swapchain.extent().width, m_swapchain.extent().height);
            continue;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            Logger::error("Swapchain acquire failed: VkResult {}", static_cast<int>(result));
            continue;
        }

        // Record tonemap: hdrImage → swapchain image.
        vkResetCommandBuffer(frame.displayCmd, 0);
        const VkCommandBufferBeginInfo beginInfo{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        if (vkBeginCommandBuffer(frame.displayCmd, &beginInfo) != VK_SUCCESS) {
            Logger::error("Failed to begin display command buffer");
            continue;
        }
        const std::array preToneMapBarriers{
            imageBarrier(m_hdrImage.handle(),
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_READ_BIT),
            imageBarrier(m_swapchain.image(imageIndex),
                         m_swapchainLayouts[imageIndex],
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_PIPELINE_STAGE_2_NONE,
                         0,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT),
        };
        pipelineBarrier(frame.displayCmd, preToneMapBarriers);

        m_toneMapper.record(frame.displayCmd,
                            m_hdrImage.view(),
                            m_swapchain.imageView(imageIndex),
                            m_swapchain.extent(),
                            m_swapchain.outputColorSpace());

        const std::array presentBarrier{
            imageBarrier(m_swapchain.image(imageIndex),
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                         VK_PIPELINE_STAGE_2_NONE,
                         0),
        };
        pipelineBarrier(frame.displayCmd, presentBarrier);

        if (vkEndCommandBuffer(frame.displayCmd) != VK_SUCCESS) {
            Logger::error("Failed to end display command buffer");
            continue;
        }

        // Submit tonemap: wait on trace timeline + imageAvailable; signal renderComplete + next timeline.
        const uint64_t displayValue = m_nextTimelineValue++;

        const std::array<VkSemaphoreSubmitInfo, 2> waitInfos{{
            {
                .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = m_timelineSemaphore,
                .value       = traceValue,
                .stageMask   = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .deviceIndex = 0,
            },
            {
                .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = frame.imageAvailable,
                .value       = 0,
                .stageMask   = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .deviceIndex = 0,
            },
        }};
        const std::array<VkSemaphoreSubmitInfo, 2> signalInfos{{
            {
                .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = m_renderComplete[imageIndex],
                .value       = 0,
                .stageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .deviceIndex = 0,
            },
            {
                .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                .pNext       = nullptr,
                .semaphore   = m_timelineSemaphore,
                .value       = displayValue,
                .stageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .deviceIndex = 0,
            },
        }};
        const VkCommandBufferSubmitInfo displayCmdInfo{
            .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext         = nullptr,
            .commandBuffer = frame.displayCmd,
            .deviceMask    = 0,
        };
        const VkSubmitInfo2 displaySubmit{
            .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext                    = nullptr,
            .flags                    = 0,
            .waitSemaphoreInfoCount   = static_cast<uint32_t>(waitInfos.size()),
            .pWaitSemaphoreInfos      = waitInfos.data(),
            .commandBufferInfoCount   = 1,
            .pCommandBufferInfos      = &displayCmdInfo,
            .signalSemaphoreInfoCount = static_cast<uint32_t>(signalInfos.size()),
            .pSignalSemaphoreInfos    = signalInfos.data(),
        };
        result = vkQueueSubmit2(m_context.deviceContext().graphicsQueue, 1, &displaySubmit, VK_NULL_HANDLE);
        if (result != VK_SUCCESS) {
            Logger::error("Display submit failed: VkResult {}", static_cast<int>(result));
            continue;
        }

        // This slot must not be reused until display has also completed.
        frame.completionValue = displayValue;

        // Present (application concern).
        result = m_swapchain.present(m_context.deviceContext().graphicsQueue, imageIndex, m_renderComplete[imageIndex]);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            handleResize(m_swapchain.extent().width, m_swapchain.extent().height);
        } else if (result != VK_SUCCESS) {
            Logger::error("Present failed: VkResult {}", static_cast<int>(result));
        }
        m_swapchainLayouts[imageIndex] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    vkDeviceWaitIdle(m_context.deviceContext().device);
    return 0;
}

uint64_t Application::renderFrame(Image& hdrTarget) {
    FrameResources& frame = m_frames[m_currentFrame];

    // Wait for the previous use of this frame slot to complete.
    if (frame.completionValue > 0U) {
        const VkSemaphoreWaitInfo waitInfo{
            .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext          = nullptr,
            .flags          = 0,
            .semaphoreCount = 1,
            .pSemaphores    = &m_timelineSemaphore,
            .pValues        = &frame.completionValue,
        };
        vkWaitSemaphores(m_context.deviceContext().device, &waitInfo, UINT64_MAX);
    }

    vkResetCommandBuffer(frame.traceCmd, 0);
    const VkCommandBufferBeginInfo beginInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    if (vkBeginCommandBuffer(frame.traceCmd, &beginInfo) != VK_SUCCESS) {
        Logger::error("Failed to begin trace command buffer");
        return frame.completionValue; // return last known safe value
    }

    if (m_pathTracer.render(frame.traceCmd, m_scene, m_camera, hdrTarget, m_gNormal, m_gDepth, m_frameIndex) != VK_SUCCESS) {
        Logger::error("PathTracer render failed");
    }

    if (vkEndCommandBuffer(frame.traceCmd) != VK_SUCCESS) {
        Logger::error("Failed to end trace command buffer");
        return frame.completionValue;
    }

    const uint64_t signalValue = m_nextTimelineValue++;
    const VkCommandBufferSubmitInfo cmdInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext         = nullptr,
        .commandBuffer = frame.traceCmd,
        .deviceMask    = 0,
    };
    const VkSemaphoreSubmitInfo timelineSignal{
        .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext       = nullptr,
        .semaphore   = m_timelineSemaphore,
        .value       = signalValue,
        .stageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0,
    };
    const VkSubmitInfo2 submitInfo{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext                    = nullptr,
        .flags                    = 0,
        .waitSemaphoreInfoCount   = 0,
        .pWaitSemaphoreInfos      = nullptr,
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos    = &timelineSignal,
    };
    if (vkQueueSubmit2(m_context.deviceContext().graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        Logger::error("Trace queue submit failed");
        return frame.completionValue;
    }

    frame.completionValue = signalValue;
    ++m_frameIndex;
    m_currentFrame = (m_currentFrame + 1U) % static_cast<uint32_t>(m_frames.size());
    return signalValue;
}

void Application::handleResize(uint32_t w, uint32_t h) {
    if (w == 0U || h == 0U || m_context.deviceContext().device == VK_NULL_HANDLE) {
        return;
    }

    vkDeviceWaitIdle(m_context.deviceContext().device);
    const VkExtent2D newExtent{
        .width = w,
        .height = h,
    };
    if (m_swapchain.recreate(newExtent) != VK_SUCCESS) {
        Logger::error("Swapchain recreate failed");
        return;
    }

    // Recreate per-image renderComplete semaphores to match the new swapchain image count.
    // vkDeviceWaitIdle() was called above, so all present operations have finished
    // and all semaphores are in the unsignaled state — safe to destroy and recreate.
    const VkDevice device = m_context.deviceContext().device;
    for (VkSemaphore& sem : m_renderComplete) {
        vkDestroySemaphore(device, sem, nullptr);
    }
    m_renderComplete.resize(m_swapchain.imageCount());
    for (VkSemaphore& sem : m_renderComplete) {
        if (createBinarySemaphore(device, sem) != VK_SUCCESS) {
            Logger::error("renderComplete semaphore recreate failed");
            return;
        }
    }

    auto hdrImage = Image::create(m_context.deviceContext(),
                                  m_swapchain.extent(),
                                  VK_FORMAT_R32G32B32A32_SFLOAT,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                  "demo.hdr");
    if (!hdrImage) {
        Logger::error("HDR image recreate failed: VkResult {}", static_cast<int>(hdrImage.error()));
        return;
    }
    m_hdrImage = std::move(*hdrImage);

    auto gNormal = Image::create(m_context.deviceContext(),
                                 m_swapchain.extent(),
                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_IMAGE_USAGE_STORAGE_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                 "demo.gNormal");
    if (!gNormal) {
        Logger::error("G-buffer normal recreate failed: VkResult {}", static_cast<int>(gNormal.error()));
        return;
    }
    m_gNormal = std::move(*gNormal);

    auto gDepth = Image::create(m_context.deviceContext(),
                                m_swapchain.extent(),
                                VK_FORMAT_R32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                "demo.gDepth");
    if (!gDepth) {
        Logger::error("G-buffer depth recreate failed: VkResult {}", static_cast<int>(gDepth.error()));
        return;
    }
    m_gDepth = std::move(*gDepth);

    auto initCmd = m_commandPool.beginOneShot();
    if (initCmd) {
        m_hdrImage.transition(*initCmd,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_GENERAL,
                              VK_PIPELINE_STAGE_2_NONE,
                              0,
                              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                              VK_ACCESS_2_SHADER_WRITE_BIT);
        m_gNormal.transition(*initCmd,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_NONE,
                             0,
                             VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_ACCESS_2_SHADER_WRITE_BIT);
        m_gDepth.transition(*initCmd,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_NONE,
                            0,
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                            VK_ACCESS_2_SHADER_WRITE_BIT);
        static_cast<void>(m_commandPool.endOneShot(*initCmd));
    }

    m_camera.setAspect(static_cast<float>(m_swapchain.extent().width) /
                       static_cast<float>(m_swapchain.extent().height));
    m_pathTracer.onResize(m_swapchain.extent());

    // Recreate tone mapper in case the swapchain format changed (e.g. HDR10 ↔ SDR).
    const std::filesystem::path resolvedShaderDir = resolveShaderDir(m_config.shaderDir);
    auto toneMapper = ToneMapper::create(m_context.deviceContext(),
                                         m_descriptors.pipelineLayout(),
                                         m_swapchain.format(),
                                         resolvedShaderDir / "tonemap_vert.spv",
                                         resolvedShaderDir / "tonemap.spv");
    if (toneMapper) {
        m_toneMapper = std::move(*toneMapper);
    } else {
        Logger::error("ToneMapper recreate failed: VkResult {}", static_cast<int>(toneMapper.error()));
    }
}

void Application::saveEXR(const std::filesystem::path& path) {
#ifndef HYPERION_HAS_OPENEXR
    Logger::warn("OpenEXR support is not enabled; cannot save {}", path.string());
#else
    if (m_context.deviceContext().device == VK_NULL_HANDLE || !m_hdrImage.isValid()) {
        return;
    }

    vkDeviceWaitIdle(m_context.deviceContext().device);
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(m_hdrImage.extent().width) *
                                  static_cast<VkDeviceSize>(m_hdrImage.extent().height) * sizeof(float) * 4U;
    auto readback = Buffer::create(m_context.deviceContext(),
                                   byteSize,
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   "demo.hdr.readback");
    if (!readback) {
        Logger::error("Failed to create readback buffer: VkResult {}", static_cast<int>(readback.error()));
        return;
    }

    auto cmd = m_commandPool.beginOneShot();
    if (!cmd) {
        Logger::error("Failed to allocate screenshot command buffer: VkResult {}", static_cast<int>(cmd.error()));
        return;
    }
    m_hdrImage.transition(*cmd,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
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
        .imageExtent = VkExtent3D{m_hdrImage.extent().width, m_hdrImage.extent().height, 1},
    };
    vkCmdCopyImageToBuffer(
        *cmd, m_hdrImage.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback->handle(), 1, &copyRegion);

    m_hdrImage.transition(*cmd,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                          VK_ACCESS_2_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                          VK_ACCESS_2_SHADER_WRITE_BIT);

    if (const VkResult result = m_commandPool.endOneShot(*cmd); result != VK_SUCCESS) {
        Logger::error("Screenshot copy failed: VkResult {}", static_cast<int>(result));
        return;
    }

    using namespace OPENEXR_IMF_NAMESPACE;
    Header header(static_cast<int>(m_hdrImage.extent().width), static_cast<int>(m_hdrImage.extent().height));
    header.channels().insert("R", Channel(FLOAT));
    header.channels().insert("G", Channel(FLOAT));
    header.channels().insert("B", Channel(FLOAT));

    FrameBuffer frameBuffer;
    char* const base = static_cast<char*>(readback->mappedData());
    const size_t pixelStride = sizeof(float) * 4U;
    const size_t rowStride = pixelStride * m_hdrImage.extent().width;
    frameBuffer.insert("R", Slice(FLOAT, base + 0U * sizeof(float), pixelStride, rowStride));
    frameBuffer.insert("G", Slice(FLOAT, base + 1U * sizeof(float), pixelStride, rowStride));
    frameBuffer.insert("B", Slice(FLOAT, base + 2U * sizeof(float), pixelStride, rowStride));

    OutputFile file(path.string().c_str(), header);
    file.setFrameBuffer(frameBuffer);
    file.writePixels(static_cast<int>(m_hdrImage.extent().height));
    Logger::info("Saved HDR screenshot to {}", path.string());
#endif
}

void Application::savePNG(const std::filesystem::path& path) {
    if (m_context.deviceContext().device == VK_NULL_HANDLE || !m_hdrImage.isValid()) {
        return;
    }

    vkDeviceWaitIdle(m_context.deviceContext().device);
    const uint32_t      width = m_hdrImage.extent().width;
    const uint32_t      height = m_hdrImage.extent().height;
    const VkDeviceSize  byteSize =
        static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * sizeof(float) * 4U;

    auto readback = Buffer::create(m_context.deviceContext(),
                                   byteSize,
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   "demo.hdr.readback.png");
    if (!readback) {
        Logger::error("savePNG: failed to create readback buffer: VkResult {}",
                      static_cast<int>(readback.error()));
        return;
    }

    auto cmd = m_commandPool.beginOneShot();
    if (!cmd) {
        Logger::error("savePNG: failed to allocate command buffer: VkResult {}",
                      static_cast<int>(cmd.error()));
        return;
    }
    m_hdrImage.transition(*cmd,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
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
        .imageExtent = VkExtent3D{width, height, 1},
    };
    vkCmdCopyImageToBuffer(
        *cmd, m_hdrImage.handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback->handle(), 1, &copyRegion);
    m_hdrImage.transition(*cmd,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_IMAGE_LAYOUT_GENERAL,
                          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                          VK_ACCESS_2_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                          VK_ACCESS_2_SHADER_WRITE_BIT);
    if (const VkResult result = m_commandPool.endOneShot(*cmd); result != VK_SUCCESS) {
        Logger::error("savePNG: copy failed: VkResult {}", static_cast<int>(result));
        return;
    }

    // Tone-map (ACES SDR: Rec.2020 linear → Rec.709 linear → sRGB 8-bit) and
    // pack into a contiguous R8G8B8 byte buffer.
    const auto*          src = static_cast<const float*>(readback->mappedData());
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 3U);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t      srcIdx = (static_cast<size_t>(y) * width + x) * 4U;
            const glm::vec3   hdr(src[srcIdx + 0], src[srcIdx + 1], src[srcIdx + 2]);
            const glm::vec3   sdrLinear = ToneMapping::acesFittedSDR(hdr);
            const glm::vec3   sdrGamma  = ColorSpace::linearRec709ToSrgb(sdrLinear);
            const glm::vec3   clamped   = glm::clamp(sdrGamma, 0.f, 1.f);
            const size_t      dstIdx    = (static_cast<size_t>(y) * width + x) * 3U;
            pixels[dstIdx + 0] = static_cast<uint8_t>(std::lround(clamped.r * 255.f));
            pixels[dstIdx + 1] = static_cast<uint8_t>(std::lround(clamped.g * 255.f));
            pixels[dstIdx + 2] = static_cast<uint8_t>(std::lround(clamped.b * 255.f));
        }
    }

    const int stride = static_cast<int>(width) * 3;
    if (!stbi_write_png(path.string().c_str(), static_cast<int>(width), static_cast<int>(height), 3,
                        pixels.data(), stride)) {
        Logger::error("savePNG: stbi_write_png failed for {}", path.string());
        return;
    }
    Logger::info("Saved tone-mapped PNG to {}", path.string());
}

void Application::destroy() noexcept {
    if (m_context.deviceContext().device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_context.deviceContext().device);
        for (FrameResources& frame : m_frames) {
            if (frame.imageAvailable != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_context.deviceContext().device, frame.imageAvailable, nullptr);
            }
            frame = {};
        }
        for (VkSemaphore& sem : m_renderComplete) {
            if (sem != VK_NULL_HANDLE)
                vkDestroySemaphore(m_context.deviceContext().device, sem, nullptr);
        }
        m_renderComplete.clear();
        if (m_timelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_context.deviceContext().device, m_timelineSemaphore, nullptr);
            m_timelineSemaphore = VK_NULL_HANDLE;
        }
    }

    m_hdrImage = {};
    m_gNormal = {};
    m_gDepth = {};
    m_iblProbe = {};
    m_pathTracer = {};
    m_toneMapper = {};
    m_sbt = {};
    m_pipeline = {};
    m_descriptors = {};
    m_scene = {};
    m_swapchain = {};
    m_commandPool = {};
    m_context = {};

    if (m_window != nullptr) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }
    SDL_Quit();

    m_swapchainLayouts.clear();
    m_running = false;
    m_nextTimelineValue = 1;
    m_currentFrame = 0;
    m_frameIndex = 0;
}
