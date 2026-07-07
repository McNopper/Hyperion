#include "demo/Application.hpp"

#include <slang-math/slang-math.hpp>

#include <array>
#include <cmath>
#include <format>
#include <utility>

#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Logger.hpp"

namespace {

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

} // namespace

int Application::run(Config config, DemoConfig demoConfig) {
    m_demoConfig = std::move(demoConfig);
    return harmonia::App::run(std::move(config));
}

bool Application::onInitialize() {
    m_positionFetchSupported = deviceContext().positionFetchSupported;
    Logger::info("VK_KHR_ray_tracing_position_fetch: {}", m_positionFetchSupported ? "enabled" : "disabled");
    Logger::info("VK_EXT_ray_tracing_invocation_reorder: {}",
                 deviceContext().serSupported ? "enabled" : "disabled");
    Logger::info("VK_KHR_ray_tracing_maintenance1 (indirect RT2): {}",
                 deviceContext().indirectRt2Supported ? "enabled" : "disabled");

    m_shaderDir = resolveShaderDir(m_demoConfig.shaderDir);
    const std::filesystem::path closestHitPath = m_positionFetchSupported ? m_shaderDir / "closesthit_pf.spv"
                                                                          : m_shaderDir / "closesthit.spv";
    const Pipeline::ShaderPaths shaderPaths{
        .raygen = m_shaderDir / "raygen.spv",
        .closesthitTriangle = closestHitPath,
        .closesthitSphere = closestHitPath,
        .intersection = m_shaderDir / "intersection.spv",
        .miss = m_shaderDir / "miss.spv",
        .shadowMiss = m_shaderDir / "shadow_miss.spv",
    };
    auto pipeline = Pipeline::create(deviceContext(), descriptors(), shaderPaths, m_demoConfig.maxDepth);
    if (!pipeline) {
        Logger::error("Pipeline creation failed: VkResult {}", static_cast<int>(pipeline.error()));
        return false;
    }
    m_pipeline = std::move(*pipeline);

    auto sbt = ShaderBindingTable::create(deviceContext(), m_pipeline, context().physicalDeviceInfo().rtProps);
    if (!sbt) {
        Logger::error("SBT creation failed: VkResult {}", static_cast<int>(sbt.error()));
        return false;
    }
    m_sbt = std::move(*sbt);

    // Progressive accumulation in the interactive window so the live preview
    // converges to the SAME image the offscreen capture path produces. Without
    // this the window shows raw single-frame path-tracer samples (and skips the
    // denoiser temporal history), which would not match the accumulated, history-
    // denoised offscreen render. The camera is static in the demo, so no
    // resetAccumulation() wiring is required; the offscreen path accumulates
    // regardless of this flag.
    setInteractiveAccumulation(true);

    return createGBuffers(swapchain().extent());
}

bool Application::createGBuffers(VkExtent2D extent) {
    auto gNormal = Image::create(deviceContext(),
                                 extent,
                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                 "demo.gNormal");
    if (!gNormal) {
        Logger::error("G-buffer normal creation failed: VkResult {}", static_cast<int>(gNormal.error()));
        return false;
    }
    m_gNormal = std::move(*gNormal);

    auto gDepth = Image::create(deviceContext(),
                                extent,
                                VK_FORMAT_R32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                "demo.gDepth");
    if (!gDepth) {
        Logger::error("G-buffer depth creation failed: VkResult {}", static_cast<int>(gDepth.error()));
        return false;
    }
    m_gDepth = std::move(*gDepth);

    m_targetsFirstUse = true;
    return true;
}

void Application::onSceneUnload() {
    m_scene = {};
    m_pathTracer = {};
}

bool Application::onSceneLoaded(const SceneLoader::SceneConfig& sceneConfig) {
    if (sceneConfig.spp && !m_demoConfig.sppExplicit) {
        m_demoConfig.spp = *sceneConfig.spp;
    }
    if (sceneConfig.maxDepth) {
        m_demoConfig.maxDepth = *sceneConfig.maxDepth;
    }
    const float envLuminance = sceneConfig.envUnitNits.value_or(1.0f);

    // Build camera — fall back to Cornell box defaults when scene file omits settings.
    Camera::PhysicalCamera physical{};
    if (sceneConfig.cameraEv100) {
        // EV100 = log2(N² × t_inv × 100/ISO) with N=1, ISO=100 → t_inv = 2^EV100
        const float ev100 = *sceneConfig.cameraEv100;
        physical.aperture = 1.0f;
        physical.iso = 100.0f;
        physical.shutterSpeedHz = std::pow(2.0f, ev100);
    }
    const sm::float3 camPos = sceneConfig.cameraPos.value_or(sm::float3(278.0f, 273.0f, -800.0f));
    const sm::float3 camAt = sceneConfig.cameraAt.value_or(sm::float3(278.0f, 273.0f, 279.5f));
    const auto [nearPlane, farPlane] = Camera::nearFarFromDistance(sm::length(camAt - camPos));
    m_camera = Camera(Camera::Params{
        .position = camPos,
        .target = camAt,
        .up = sceneConfig.cameraUp.value_or(sm::float3(0.0f, 1.0f, 0.0f)),
        .vfovDeg = sceneConfig.cameraVfov.value_or(39.1f),
        .aspectRatio = static_cast<float>(swapchain().extent().width) / static_cast<float>(swapchain().extent().height),
        .nearPlane = nearPlane,
        .farPlane = farPlane,
        .lensRadius = 0.0f,
        .focusDist = 1079.5f,
        .physical = physical,
    });

    if (const VkResult result = m_scene.build(deviceContext(), commandPool()); result != VK_SUCCESS) {
        Logger::error("Scene build failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    Logger::info("Scene built (BLAS+TLAS)");

    if (const VkResult result = descriptors().updateSceneSet(deviceContext(),
                                                             m_scene.instanceBuffer().handle(),
                                                             m_scene.materialBuffer().handle(),
                                                             m_scene.vertexBuffer().handle(),
                                                             m_scene.indexBuffer().handle(),
                                                             m_scene.lightBuffer().handle(),
                                                             m_scene.emissiveTriangleBuffer().handle(),
                                                             m_scene.emissiveCdfBuffer().handle(),
                                                             m_scene.textures());
        result != VK_SUCCESS) {
        Logger::error("Descriptor update failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    Logger::info("Descriptors updated");

    const bool hasIbl = iblProbe().has_value() && iblProbe()->isValid();
    auto tracer = PathTracer::create(deviceContext(),
                                     swapchain().extent(),
                                     m_pipeline,
                                     m_sbt,
                                     descriptors(),
                                     PathTracer::Config{
                                         .samplesPerPixel = m_demoConfig.spp,
                                         .maxDepth = m_demoConfig.maxDepth,
                                         .envLuminance = envLuminance,
                                         .hasEnvMap = hasIbl ? 1u : 0u,
                                         .envImportanceWidth = hasIbl ? iblProbe()->cdfWidth() : 0u,
                                         .envImportanceHeight = hasIbl ? iblProbe()->cdfHeight() : 0u,
                                         .tonemapper = tonemapper(),
                                         .workingColorSpace = static_cast<uint32_t>(workingColorSpace()),
                                         .serEnabled = deviceContext().serSupported,
                                         .indirectRt2Enabled = deviceContext().indirectRt2Supported,
                                     });
    if (!tracer) {
        Logger::error("PathTracer creation failed: VkResult {}", static_cast<int>(tracer.error()));
        return false;
    }
    m_pathTracer = std::move(*tracer);
    Logger::info("PathTracer created");

    m_targetsFirstUse = true;
    return true;
}

void Application::record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept {
    static_cast<void>(target);
    if (m_targetsFirstUse) {
        // The host hands the HDR target over with undefined contents; the
        // renderer owns its layout — bring everything to GENERAL once.
        const std::array barriers{
            harmonia::imageBarrier(hdrImage().handle(),
                                   VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_NONE,
                                   0,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_ACCESS_2_SHADER_WRITE_BIT),
            harmonia::imageBarrier(m_gNormal.handle(),
                                   VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_NONE,
                                   0,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_ACCESS_2_SHADER_WRITE_BIT),
            harmonia::imageBarrier(m_gDepth.handle(),
                                   VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_2_NONE,
                                   0,
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                   VK_ACCESS_2_SHADER_WRITE_BIT),
        };
        harmonia::pipelineBarrier(cmd, barriers);
        m_targetsFirstUse = false;
    }

    if (m_pathTracer.render(cmd, m_scene, m_camera, hdrImage(), m_gNormal, m_gDepth, frameIndex()) != VK_SUCCESS) {
        Logger::error("PathTracer render failed");
    }
}

void Application::onResize(VkExtent2D extent) noexcept {
    if (!createGBuffers(extent)) {
        return;
    }
    m_camera.setAspect(static_cast<float>(extent.width) / static_cast<float>(extent.height));
    m_pathTracer.onResize(extent);
}

void Application::onResized(VkExtent2D extent) {
    static_cast<void>(extent);
}

bool Application::onEvent(const SDL_Event& event) {
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_S) {
        static_cast<void>(saveExr(std::format("hyperion_{:06}.exr", frameIndex())));
        return true;
    }
    return false;
}
