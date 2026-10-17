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
        harmonia::core::Logger::info("Using shader dir (user-specified): {}", shaderDir.string());
        return shaderDir;
    }
    // Prefer the compile-time output dir (build/shaders) over any source-tree
    // "shaders/" that lacks .spv files.
#ifdef HYPERION_SHADER_DIR
    std::filesystem::path builtDir = HYPERION_SHADER_DIR;
    if (std::filesystem::exists(builtDir)) {
        harmonia::core::Logger::info("Using shader dir (built): {}", builtDir.string());
        return builtDir;
    }
#endif
    harmonia::core::Logger::error("No shader directory found (tried '{}' and HYPERION_SHADER_DIR)", shaderDir.string());
    return shaderDir;
}

} // namespace

int hyperion::Application::run(Config&& config, DemoConfig&& demoConfig) {
    m_demoConfig = std::move(demoConfig);
    return harmonia::app::App::run(std::move(config));
}

bool hyperion::Application::onInitialize() {
    m_positionFetchSupported = deviceContext().positionFetchSupported;
    harmonia::core::Logger::info("VK_KHR_ray_tracing_position_fetch: {}",
                                 m_positionFetchSupported ? "enabled" : "disabled");
    harmonia::core::Logger::info("VK_EXT_ray_tracing_invocation_reorder: {}",
                                 deviceContext().serSupported ? "enabled" : "disabled");
    harmonia::core::Logger::info("VK_KHR_ray_tracing_maintenance1 (indirect RT2): required");

    m_shaderDir = resolveShaderDir(m_demoConfig.shaderDir);
    harmonia::renderer::Pipeline::ShaderPaths shaderPaths = hyperion::makeHyperionShaderPaths(m_shaderDir);
    if (m_positionFetchSupported) {
        const std::filesystem::path closestHitPath = m_shaderDir / "closesthit_pf.spv";
        shaderPaths.closesthitTriangle = closestHitPath;
        shaderPaths.closesthitSphere = closestHitPath;
    }
    // RT pipeline recursion depth is a limit on the shader's TraceRay NESTING, not the
    // path bounce budget — the two are unrelated. Hyperion nests exactly two deep:
    // raygen -> TraceRay -> closesthit, and closesthit -> TraceRay (shadow) ->
    // anyhit/shadowMiss, where RAY_FLAG_SKIP_CLOSEST_HIT_SHADER stops that shadow ray
    // from recursing further. The depth is therefore a constant 2 for any maxDepth.
    //
    // Passing m_demoConfig.maxDepth here conflated the two concepts and only worked by
    // accident: onInitialize() runs before applySceneOverrides(), so the pipeline was
    // always built with the pre-override default (8). Had the scene override landed
    // first, a max_depth=1 scene would have declared a recursion limit of 1 and broken
    // every NEE shadow ray.
    constexpr std::uint32_t kRayRecursionDepth = 2;
    auto pipeline =
        harmonia::renderer::Pipeline::create(deviceContext(), descriptors(), shaderPaths, kRayRecursionDepth);
    if (!pipeline) {
        harmonia::core::Logger::error("harmonia::renderer::Pipeline creation failed: VkResult {}",
                                      static_cast<int>(pipeline.error()));
        return false;
    }
    m_pipeline = std::move(*pipeline);

    auto sbt =
        hyperion::ShaderBindingTable::create(deviceContext(), m_pipeline, context().physicalDeviceInfo().rtProps);
    if (!sbt) {
        harmonia::core::Logger::error("SBT creation failed: VkResult {}", static_cast<int>(sbt.error()));
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

bool hyperion::Application::createGBuffers(VkExtent2D extent) {
    auto gNormal = harmonia::core::Image::create(deviceContext(),
                                                 extent,
                                                 VK_FORMAT_R16G16B16A16_SFLOAT,
                                                 VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 "demo.gNormal");
    if (!gNormal) {
        harmonia::core::Logger::error("G-buffer normal creation failed: VkResult {}",
                                      static_cast<int>(gNormal.error()));
        return false;
    }
    m_gNormal = std::move(*gNormal);

    auto gDepth = harmonia::core::Image::create(deviceContext(),
                                                extent,
                                                VK_FORMAT_R32_SFLOAT,
                                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                "demo.gDepth");
    if (!gDepth) {
        harmonia::core::Logger::error("G-buffer depth creation failed: VkResult {}", static_cast<int>(gDepth.error()));
        return false;
    }
    m_gDepth = std::move(*gDepth);

    m_targetsFirstUse = true;
    return true;
}

void hyperion::Application::onSceneUnload() {
    m_scene = {};
    m_pathTracer = {};
}

bool hyperion::Application::onSceneLoaded(const harmonia::scene::SceneLoader::SceneConfig& sceneConfig) {
    applySceneOverrides(sceneConfig);
    const float envLuminance = sceneConfig.envUnitNits.value_or(1.0f);
    buildCamera(sceneConfig);

    if (const VkResult result = m_scene.build(deviceContext(), commandPool()); result != VK_SUCCESS) {
        harmonia::core::Logger::error("hyperion::Scene build failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    harmonia::core::Logger::info("hyperion::Scene built (BLAS+TLAS)");

    if (const VkResult result = setupSceneDescriptors(); result != VK_SUCCESS) {
        harmonia::core::Logger::error("Descriptor update failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    harmonia::core::Logger::info("harmonia::renderer::Descriptors updated");

    const auto& probe = iblProbe();
    const bool hasIbl = probe.has_value() && probe->isValid();
    auto tracer = hyperion::PathTracer::create(
        deviceContext(),
        swapchain().extent(),
        m_pipeline,
        m_sbt,
        descriptors(),
        hyperion::PathTracer::Config{.samplesPerPixel = m_demoConfig.spp,
                                     .maxDepth = m_demoConfig.maxDepth,
                                     .envLuminance = envLuminance,
                                     .hasEnvMap = hasIbl ? 1u : 0u,
                                     .envImportanceWidth = (probe && hasIbl) ? probe->cdfWidth() : 0u,
                                     .envImportanceHeight = (probe && hasIbl) ? probe->cdfHeight() : 0u,
                                     .tonemapper = static_cast<harmonia::Tonemapper>(tonemapper()),
                                     .workingColorSpace = workingColorSpace(),
                                     .serEnabled = deviceContext().serSupported});
    if (!tracer) {
        harmonia::core::Logger::error("hyperion::PathTracer creation failed: VkResult {}",
                                      static_cast<int>(tracer.error()));
        return false;
    }
    m_pathTracer = std::move(*tracer);
    harmonia::core::Logger::info("hyperion::PathTracer created");

    m_targetsFirstUse = true;
    return true;
}

void hyperion::Application::applySceneOverrides(const harmonia::scene::SceneLoader::SceneConfig& config) {
    if (config.spp && !m_demoConfig.sppExplicit) {
        m_demoConfig.spp = *config.spp;
    }
    if (config.maxDepth && !m_demoConfig.maxDepthExplicit) {
        m_demoConfig.maxDepth = *config.maxDepth;
    }
}

void hyperion::Application::buildCamera(const harmonia::scene::SceneLoader::SceneConfig& config) {
    // Build camera — fall back to Cornell box defaults when scene file omits settings.
    harmonia::renderer::Camera::PhysicalCamera physical{};
    if (config.cameraEv100) {
        physical = harmonia::renderer::Camera::PhysicalCamera::fromEv100(*config.cameraEv100);
    }
    const sm::float3 camPos = config.cameraPos.value_or(kCornellCamPos);
    const sm::float3 camAt = config.cameraAt.value_or(kCornellCamTarget);
    const auto [nearPlane, farPlane] = harmonia::renderer::Camera::nearFarFromDistance(sm::length(camAt - camPos));
    m_camera = harmonia::renderer::Camera(harmonia::renderer::Camera::Params{
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

VkResult hyperion::Application::setupSceneDescriptors() {
    return descriptors().updateSceneSet(deviceContext(),
                                        m_scene.instanceBuffer(),
                                        m_scene.materialBuffer(),
                                        m_scene.vertexBuffer(),
                                        m_scene.indexBuffer(),
                                        m_scene.lightBuffer(),
                                        m_scene.emissiveTriangleBuffer(),
                                        m_scene.emissiveCdfBuffer(),
                                        m_scene.textures());
}

void hyperion::Application::record(VkCommandBuffer cmd, const harmonia::app::RenderTarget& target) noexcept {
    static_cast<void>(target);
    if (m_targetsFirstUse) {
        transitionTargetsOnFirstUse(cmd);
    }

    if (const VkResult r = m_pathTracer.render(cmd, m_scene, m_camera, hdrImage(), m_gNormal, m_gDepth, frameIndex());
        r != VK_SUCCESS) {
        harmonia::core::Logger::error("hyperion::PathTracer render failed: VkResult {}", static_cast<std::int32_t>(r));
    }
}

void hyperion::Application::transitionTargetsOnFirstUse(VkCommandBuffer cmd) noexcept {
    // The host hands the HDR target over with undefined contents; the
    // renderer owns its layout — bring everything to GENERAL once.
    const std::array barriers{
        harmonia::core::imageBarrier(hdrImage().handle(),
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     0,
                                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                     VK_ACCESS_2_SHADER_WRITE_BIT),
        harmonia::core::imageBarrier(m_gNormal.handle(),
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     0,
                                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                     VK_ACCESS_2_SHADER_WRITE_BIT),
        harmonia::core::imageBarrier(m_gDepth.handle(),
                                     VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     VK_PIPELINE_STAGE_2_NONE,
                                     0,
                                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                                     VK_ACCESS_2_SHADER_WRITE_BIT),
    };
    harmonia::core::pipelineBarrier(cmd, barriers);
    m_targetsFirstUse = false;
}

void hyperion::Application::onResize(VkExtent2D extent) noexcept {
    if (!createGBuffers(extent)) {
        return;
    }
    m_camera.setAspect(static_cast<float>(extent.width) / static_cast<float>(extent.height));
    m_pathTracer.onResize(extent);
}

bool hyperion::Application::onEvent(const SDL_Event& event) {
    if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_S) {
        static_cast<void>(saveExr(std::format("hyperion_{:06}.exr", frameIndex())));
        return true;
    }
    return false;
}
