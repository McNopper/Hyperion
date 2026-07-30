#include "demo/Application.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <slang-math/slang-math.hpp>
#include <utility>

#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Logger.hpp"
#include "hyperion/ShaderPaths.hpp"

namespace {

constexpr sm::float3 kCornellCamPos(278.0f, 273.0f, -800.0f);
constexpr sm::float3 kCornellCamTarget(278.0f, 273.0f, 279.5f);
constexpr float kCornellVfovDeg = 39.1f;
constexpr float kCornellFocusDist = 1079.5f;

[[nodiscard]] std::filesystem::path resolveShaderDir(std::filesystem::path shaderDir) {
    // Explicit user path takes priority only if it actually exists.
    if (!shaderDir.empty() && std::filesystem::exists(shaderDir)) {
        harmonia::Logger::info("Using shader dir (user-specified): {}", shaderDir.string());
        return shaderDir;
    }
    // Prefer the compile-time output dir (build/shaders) over any source-tree
    // "shaders/" that lacks .spv files.
#ifdef HYPERION_SHADER_DIR
    std::filesystem::path builtDir = HYPERION_SHADER_DIR;
    if (std::filesystem::exists(builtDir)) {
        harmonia::Logger::info("Using shader dir (built): {}", builtDir.string());
        return builtDir;
    }
#endif
    harmonia::Logger::error("No shader directory found (tried '{}' and HYPERION_SHADER_DIR)", shaderDir.string());
    return shaderDir;
}

} // namespace

int Application::run(Config&& config, DemoConfig&& demoConfig) {
    m_demoConfig = std::move(demoConfig);
    return harmonia::App::run(std::move(config));
}

bool Application::onInitialize() {
    m_positionFetchSupported = deviceContext().positionFetchSupported;
    harmonia::Logger::info("VK_KHR_ray_tracing_position_fetch: {}", m_positionFetchSupported ? "enabled" : "disabled");
    harmonia::Logger::info("VK_EXT_ray_tracing_invocation_reorder: {}", deviceContext().serSupported ? "enabled" : "disabled");
    harmonia::Logger::info("VK_KHR_ray_tracing_maintenance1 (indirect RT2): {}",
                 deviceContext().indirectRt2Supported ? "enabled" : "disabled");

    m_shaderDir = resolveShaderDir(m_demoConfig.shaderDir);
    harmonia::Pipeline::ShaderPaths shaderPaths = makeHyperionShaderPaths(m_shaderDir);
    if (m_positionFetchSupported) {
        const std::filesystem::path closestHitPath = m_shaderDir / "closesthit_pf.spv";
        shaderPaths.closesthitTriangle = closestHitPath;
        shaderPaths.closesthitSphere = closestHitPath;
    }
    auto pipeline = harmonia::Pipeline::create(deviceContext(), descriptors(), shaderPaths, m_demoConfig.maxDepth);
    if (!pipeline) {
        harmonia::Logger::error("harmonia::Pipeline creation failed: VkResult {}", static_cast<int>(pipeline.error()));
        return false;
    }
    m_pipeline = std::move(*pipeline);

    auto sbt = ShaderBindingTable::create(deviceContext(), m_pipeline, context().physicalDeviceInfo().rtProps);
    if (!sbt) {
        harmonia::Logger::error("SBT creation failed: VkResult {}", static_cast<int>(sbt.error()));
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
    auto gNormal = harmonia::Image::create(deviceContext(),
                                 extent,
                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                 "demo.gNormal");
    if (!gNormal) {
        harmonia::Logger::error("G-buffer normal creation failed: VkResult {}", static_cast<int>(gNormal.error()));
        return false;
    }
    m_gNormal = std::move(*gNormal);

    auto gDepth = harmonia::Image::create(deviceContext(),
                                extent,
                                VK_FORMAT_R32_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                "demo.gDepth");
    if (!gDepth) {
        harmonia::Logger::error("G-buffer depth creation failed: VkResult {}", static_cast<int>(gDepth.error()));
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

bool Application::onSceneLoaded(const harmonia::SceneLoader::SceneConfig& sceneConfig) {
    applySceneOverrides(sceneConfig);
    const float envLuminance = sceneConfig.envUnitNits.value_or(1.0f);
    buildCamera(sceneConfig);

    if (const VkResult result = m_scene.build(deviceContext(), commandPool()); result != VK_SUCCESS) {
        harmonia::Logger::error("Scene build failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    harmonia::Logger::info("Scene built (BLAS+TLAS)");

    if (const VkResult result = setupSceneDescriptors(); result != VK_SUCCESS) {
        harmonia::Logger::error("Descriptor update failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    harmonia::Logger::info("harmonia::Descriptors updated");

    const auto& probe = iblProbe();
    const bool hasIbl = probe.has_value() && probe->isValid();
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
                                         .envImportanceWidth = (probe && hasIbl) ? probe->cdfWidth() : 0u,
                                         .envImportanceHeight = (probe && hasIbl) ? probe->cdfHeight() : 0u,
                                         .tonemapper = static_cast<harmonia::Tonemapper>(tonemapper()),
                                         .workingColorSpace = workingColorSpace(),
                                         .serEnabled = deviceContext().serSupported,
                                         .indirectRt2Enabled = deviceContext().indirectRt2Supported,
                                     });
    if (!tracer) {
        harmonia::Logger::error("PathTracer creation failed: VkResult {}", static_cast<int>(tracer.error()));
        return false;
    }
    m_pathTracer = std::move(*tracer);
    harmonia::Logger::info("PathTracer created");

    m_targetsFirstUse = true;
    return true;
}

void Application::applySceneOverrides(const harmonia::SceneLoader::SceneConfig& config) {
    if (config.spp && !m_demoConfig.sppExplicit) {
        m_demoConfig.spp = *config.spp;
    }
    if (config.maxDepth) {
        m_demoConfig.maxDepth = *config.maxDepth;
    }
}

void Application::buildCamera(const harmonia::SceneLoader::SceneConfig& config) {
    // Build camera — fall back to Cornell box defaults when scene file omits settings.
    harmonia::Camera::PhysicalCamera physical{};
    if (config.cameraEv100) {
        physical = harmonia::Camera::PhysicalCamera::fromEv100(*config.cameraEv100);
    }
    const sm::float3 camPos = config.cameraPos.value_or(kCornellCamPos);
    const sm::float3 camAt = config.cameraAt.value_or(kCornellCamTarget);
    const auto [nearPlane, farPlane] = harmonia::Camera::nearFarFromDistance(sm::length(camAt - camPos));
    m_camera = harmonia::Camera(harmonia::Camera::Params{
        .position = camPos,
        .target = camAt,
        .up = config.cameraUp.value_or(sm::float3(0.0f, 1.0f, 0.0f)),
        .vfovDeg = config.cameraVfov.value_or(kCornellVfovDeg),
        .aspectRatio = static_cast<float>(swapchain().extent().width) / static_cast<float>(swapchain().extent().height),
        .nearPlane = nearPlane,
        .farPlane = farPlane,
        .lensRadius = 0.0f,
        .focusDist = kCornellFocusDist,
        .physical = physical,
    });
}

VkResult Application::setupSceneDescriptors() {
    return descriptors().updateSceneSet(deviceContext(),
                                        m_scene.instanceBuffer().handle(),
                                        m_scene.materialBuffer().handle(),
                                        m_scene.vertexBuffer().handle(),
                                        m_scene.indexBuffer().handle(),
                                        m_scene.lightBuffer().handle(),
                                        m_scene.emissiveTriangleBuffer().handle(),
                                        m_scene.emissiveCdfBuffer().handle(),
                                        m_scene.textures());
}

void Application::record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept {
    static_cast<void>(target);
    if (m_targetsFirstUse) {
        transitionTargetsOnFirstUse(cmd);
    }

    if (const VkResult r = m_pathTracer.render(cmd, m_scene, m_camera, hdrImage(), m_gNormal, m_gDepth, frameIndex());
        r != VK_SUCCESS) {
        harmonia::Logger::error("PathTracer render failed: VkResult {}", static_cast<std::int32_t>(r));
    }
}

void Application::transitionTargetsOnFirstUse(VkCommandBuffer cmd) noexcept {
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

void Application::onResize(VkExtent2D extent) noexcept {
    if (!createGBuffers(extent)) {
        return;
    }
    m_camera.setAspect(static_cast<float>(extent.width) / static_cast<float>(extent.height));
    m_pathTracer.onResize(extent);
}

bool Application::onEvent(const SDL_Event& event) {
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_S) {
        static_cast<void>(saveExr(std::format("hyperion_{:06}.exr", frameIndex())));
        return true;
    }
    return false;
}
