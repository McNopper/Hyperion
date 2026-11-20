#include "demo/Application.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <slang-math/slang-math.hpp>
#include <utility>

#include "aether/format/SceneAnimator.hpp"
#include "harmonia/core/Barrier.hpp"
#include "harmonia/core/Logger.hpp"
#include "hyperion/ShaderPaths.hpp"

namespace {

constexpr sm::float3 kCornellCamPos(278.0f, 273.0f, -800.0f);
constexpr sm::float3 kCornellCamTarget(278.0f, 273.0f, 279.5f);
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
    // ANI4+ANI5: the pass's descriptors bind the previous scene's buffers.
    m_deformPass.reset();
    m_scene = {};
    m_pathTracer = {};
}

bool hyperion::Application::onSceneLoaded(const harmonia::scene::SceneLoader::SceneConfig& sceneConfig) {
    applySceneOverrides(sceneConfig);
    if (!config().outputFile.empty() && samplingPlan(m_demoConfig.spp, true).frameCount > 1U &&
        !config().stages.accumulation) {
        harmonia::core::Logger::error("Hyperion multi-sample captures require the accumulation stage");
        return false;
    }
    const float envLuminance = sceneConfig.envUnitNits.value_or(1.0f);
    // ANI8 camera-from-node: remember the scripted node; placement itself runs
    // in record()'s per-tick block (never-placed sentinel fires on frame 0).
    m_cameraNodeId = sceneConfig.cameraNodeId;
    m_cameraTick = ~std::uint64_t{0};
    buildCamera(sceneConfig);

    if (const VkResult result = m_scene.build(deviceContext(), commandPool()); result != VK_SUCCESS) {
        harmonia::core::Logger::error("hyperion::Scene build failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    harmonia::core::Logger::info("hyperion::Scene built (BLAS+TLAS)");

    // ANI4+ANI5 renderer half: the shared GPU deformation pass (skinned or
    // morphed scenes only; the rest/active vertex + deformation buffers must
    // exist by now).
    if (m_scene.hasSkins() || m_scene.hasMorphs()) {
        auto deformPass =
            harmonia::scene::DeformPass::create(deviceContext(),
                                                m_scene,
                                                m_scene.vertexRestBuffer(),
                                                m_scene.vertexBuffer(),
                                                std::filesystem::path(HARMONIA_SHADER_DIR) / "deform.comp.spv");
        if (!deformPass) {
            harmonia::core::Logger::error("DeformPass creation failed: VkResult {}",
                                          static_cast<std::int32_t>(deformPass.error()));
            return false;
        }
        m_deformPass = std::move(*deformPass);
    } else {
        m_deformPass.reset();
    }
    if (const VkResult result = setupSceneDescriptors(); result != VK_SUCCESS) {
        harmonia::core::Logger::error("Descriptor update failed: VkResult {}", static_cast<int>(result));
        return false;
    }
    harmonia::core::Logger::info("harmonia::renderer::Descriptors updated");

    const auto& probe = iblProbe();
    const bool hasIbl = probe.has_value() && probe->isValid();
    const auto samples = samplingPlan(m_demoConfig.spp, !config().outputFile.empty());
    harmonia::core::Logger::info("Sampling plan: {} frame(s), {} sample(s)/dispatch (requested spp {})",
                                 samples.frameCount,
                                 samples.samplesPerDispatch,
                                 m_demoConfig.spp);
    if (!m_demoConfig.smsEnabled) {
        harmonia::core::Logger::info("SMS mode off — ordinary path-traced reference (no SMS work or priors)");
    }
    auto tracer = hyperion::PathTracer::create(
        deviceContext(),
        swapchain().extent(),
        m_pipeline,
        m_sbt,
        descriptors(),
        hyperion::PathTracer::Config{.samplesPerPixel = samples.samplesPerDispatch,
                                     .maxDepth = m_demoConfig.maxDepth,
                                     .rngSeed = config().rngSeed,
                                     .smsEnabled = m_demoConfig.smsEnabled,
                                     .envLuminance = envLuminance,
                                     .hasEnvMap = hasIbl ? 1u : 0u,
                                     .envImportanceWidth = (probe && hasIbl) ? probe->cdfWidth() : 0u,
                                     .envImportanceHeight = (probe && hasIbl) ? probe->cdfHeight() : 0u,
                                     .tonemapper = static_cast<harmonia::Tonemapper>(tonemapper()),
                                     .workingColorSpace = workingColorSpace(),
                                     .serEnabled = deviceContext().serSupported},
        m_shaderDir);
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
        // VFOV fallback hoisted to harmonia::renderer::kDefaultVfovDeg (cross-renderer
        // audit, 2026-11-19): ONE definition shared with Theia, so the two renderers
        // cannot drift. Was kCornellVfovDeg = 39.1f here vs value_or(39.1f) in Theia.
        .vfovDeg = config.cameraVfov.value_or(harmonia::renderer::kDefaultVfovDeg),
        .aspectRatio = static_cast<float>(swapchain().extent().width) / static_cast<float>(swapchain().extent().height),
        .nearPlane = nearPlane,
        .farPlane = farPlane,
        .lensRadius = 0.0f,
        .focusDist = kCornellFocusDist,
        .physical = physical,
    });
}

void hyperion::Application::placeBoundCamera(double seconds) {
    if (!m_cameraNodeId || m_scene.attachedScene() == nullptr) {
        return;
    }
    const auto pose = aether::SceneAnimator::evaluateCamera(*m_scene.attachedScene(), static_cast<float>(seconds));
    if (!pose) {
        return; // hand-built scene without a placement - keep the current view
    }
    // aether::Vec3 is sm::float3; forward = rotation * (0, 0, -1) matches the
    // static [camera] convention SceneLoader derives At/Up with.
    const sm::float3 at = pose->position + (pose->rotation * sm::float3{0.0f, 0.0f, -1.0f});
    m_camera.setPosition(pose->position);
    m_camera.setTarget(at);
    m_camera.setUp(pose->rotation * sm::float3{0.0f, 1.0f, 0.0f});
    // Scene-scale-aware clip planes follow the placed camera-to-target
    // distance (the same helper buildCamera uses).
    const auto [nearPlane, farPlane] = harmonia::renderer::Camera::nearFarFromDistance(sm::length(at - pose->position));
    m_camera.setNearFar(nearPlane, farPlane);
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

    // PERF3 (ANI2 seam): drive the shared TLAS update for animated scenes.
    // Deterministic tick clock (Aether ANI2): seconds = tick / fps - the same
    // logical frame always renders the same scene state. The tick is
    // animationFrameIndex(): the raw frame index normally, PINNED to the output
    // frame during ANI7 sequence renders, so every accumulation sample of output
    // frame k renders the same scene state. The update refits in place (periodic
    // full rebuild) via the shared harmonia::scene_rt policy; a changed tick
    // invalidates the accumulation history (resetAccumulation) - in a sequence
    // render exactly once per output frame; in the classic paths once per frame
    // as before (the index changes every frame there).
    // ANI4 renderer half: skin-carrying scenes enter the same tick block (a
    // skin-only scene has no tracks but its POSE changes with the tick): the
    // deformation runs after the stamp and before the TLAS update.
    // ONE animation system (ANI7/ANI8): a single clock (seconds + change key)
    // drives EVERY animation consumer - scene stamping, deformation, TLAS
    // refit and the scripted camera - so they can never disagree (interval
    // captures sample time between integer ticks). Both guards below read the
    // SAME clock key; the camera keeps its own guard purely for reset
    // semantics: a camera bound to a STATIC node on a static scene must not
    // trip the animation block's accumulation reset (offline would never
    // converge) - that case re-places the camera idempotently instead.
    const harmonia::app::App::AnimationClock clock = animationClock(m_scene.animationFps());
    if (m_cameraNodeId && clock.key != m_cameraTick) {
        m_cameraTick = clock.key;
        placeBoundCamera(clock.seconds);
    }
    if (m_scene.hasAnimation() || m_scene.hasSkins() || m_scene.hasMorphs()) {
        if (clock.key != m_lastAnimationTick) {
            m_lastAnimationTick = clock.key;
            const float seconds = static_cast<float>(clock.seconds);
            if (m_scene.hasAnimation()) {
                m_scene.setAnimationTime(seconds);
            }
            if (m_deformPass.has_value()) {
                if (const VkResult deformResult = m_deformPass->execute(deviceContext(),
                                                                        commandPool(),
                                                                        m_scene,
                                                                        seconds,
                                                                        m_scene.vertexRestBuffer(),
                                                                        m_scene.vertexBuffer(),
                                                                        m_scene.meshSceneVertexOffsets());
                    deformResult != VK_SUCCESS) {
                    harmonia::core::Logger::error("Vertex deform failed: VkResult {}",
                                                  static_cast<std::int32_t>(deformResult));
                }
            }
            if (const VkResult r = m_scene.updateTlas(deviceContext(), commandPool()); r != VK_SUCCESS) {
                harmonia::core::Logger::error("TLAS update failed: VkResult {}", static_cast<std::int32_t>(r));
            }
            resetAccumulation();
            if (!config().outputFile.empty() && config().sequenceFrames == 0U && !m_animatedCaptureWarned) {
                m_animatedCaptureWarned = true;
                harmonia::core::Logger::warn(
                    "animated scene captured with --output but no --sequence-frames: accumulation restarts every "
                    "sample, so the capture is a single-sample last-tick image - use --sequence-frames <N> for "
                    "per-frame EXR/PNG output (ANI7)");
            }
        }
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
