#ifndef DEMO_APPLICATION_HPP
#define DEMO_APPLICATION_HPP

#include <volk/volk.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "harmonia/app/App.hpp"
#include "harmonia/app/IRenderer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "harmonia/scene/DeformPass.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/SamplingPlan.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

/// Hyperion demo: the path tracer injected into the shared harmonia::app::App host.
///
/// The host owns window/context/swapchain/HDR image/tonemap/present; this
/// class owns only what is path-tracer specific — RT pipeline + SBT,
/// G-buffers, the hyperion::Scene, the camera and the hyperion::PathTracer itself.  Per the host
/// contract, record() produces a linear image in the scene-referred working
/// color space and leaves it in VK_IMAGE_LAYOUT_GENERAL.
namespace hyperion {
class Application final : public harmonia::app::App,
                          public harmonia::app::IRenderer,
                          public harmonia::app::IFrameCompletion {
  public:
    struct DemoConfig {
        std::uint32_t spp = 4;
        std::uint32_t maxDepth = 8;
        bool sppExplicit = false;      ///< true if --spp was given on the command line
        bool maxDepthExplicit = false; ///< true if --depth was given on the command line
        /// GI-SMS #2 application mode: false selects the ordinary-PT reference
        /// (--sms-mode off). Spatial reuse is Theia-only and rejected at the CLI.
        bool smsEnabled = true;
        std::filesystem::path shaderDir;
    };

    int run(Config&& config, DemoConfig&& demoConfig);

    // harmonia::app::IRenderer
    void record(VkCommandBuffer cmd, const harmonia::app::RenderTarget& target) noexcept override;
    void onResize(VkExtent2D extent) noexcept override;
    [[nodiscard]] VkPipelineStageFlags2 outputStageMask() const noexcept override {
        return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
    [[nodiscard]] VkImageView gNormalView() const noexcept override { return m_gNormal.view(); }
    [[nodiscard]] VkImageView gDepthView() const noexcept override { return m_gDepth.view(); }
    [[nodiscard]] const char* name() const noexcept override { return "Hyperion hyperion::PathTracer"; }
    [[nodiscard]] harmonia::app::IFrameCompletion* frameCompletion() noexcept override { return this; }
    [[nodiscard]] std::expected<bool, VkResult> ready() noexcept override { return m_pathTracer.frameReady(); }
    [[nodiscard]] VkResult recordContinuation(VkCommandBuffer cmd) noexcept override {
        return m_pathTracer.recordContinuation(cmd);
    }

  protected:
    // harmonia::app::App hooks
    [[nodiscard]] harmonia::app::IRenderer& renderer() noexcept override { return *this; }
    [[nodiscard]] harmonia::scene::ISceneBuilder& sceneBuilder() noexcept override { return m_scene; }
    [[nodiscard]] bool onInitialize() override;
    [[nodiscard]] bool onSceneLoaded(const harmonia::scene::SceneLoader::SceneConfig& sceneConfig) override;
    void onSceneUnload() override;
    bool onEvent(const SDL_Event& event) override;
    [[nodiscard]] std::uint32_t offscreenFrameCount() const noexcept override {
        return samplingPlan(m_demoConfig.spp, true).frameCount;
    }

  private:
    [[nodiscard]] bool createGBuffers(VkExtent2D extent);
    void applySceneOverrides(const harmonia::scene::SceneLoader::SceneConfig& config);
    void buildCamera(const harmonia::scene::SceneLoader::SceneConfig& config);
    /// ANI8: re-place m_camera from the bound node's evaluated world at
    /// @p tick (position/target/up + scene-scale clip planes). No-op without
    /// a bound camera node.
    void placeBoundCamera(std::uint32_t tick);
    [[nodiscard]] VkResult setupSceneDescriptors();
    void transitionTargetsOnFirstUse(VkCommandBuffer cmd) noexcept;

    DemoConfig m_demoConfig{};
    std::filesystem::path m_shaderDir;
    harmonia::renderer::Pipeline m_pipeline{};
    hyperion::ShaderBindingTable m_sbt{};
    hyperion::PathTracer m_pathTracer{};
    hyperion::Scene m_scene{};
    harmonia::renderer::Camera m_camera{};
    /// ANI8 camera-from-node: the scripted camera node from `[camera] node`
    /// (std::nullopt = the classic static camera) plus the last tick it was
    /// placed at (never-placed sentinel forces frame 0).
    std::optional<std::string> m_cameraNodeId;
    std::uint32_t m_cameraTick = ~std::uint32_t{0};
    harmonia::core::Image m_gNormal{};     ///< G-buffer world-space normal (R16G16B16A16_SFLOAT)
    harmonia::core::Image m_gDepth{};      ///< G-buffer ray hit distance  (R32_SFLOAT)
    bool m_targetsFirstUse = true;         ///< HDR/G-buffer images need UNDEFINED→GENERAL
    bool m_positionFetchSupported = false; ///< VK_KHR_ray_tracing_position_fetch is active.
    /// ANI7: last animation tick applied - the guard that stamps + resets ONCE per
    /// OUTPUT frame during sequence renders (per accumulation sample otherwise, where
    /// the raw index changes every frame anyway). ~0u = "never" so frame 0 always runs.
    std::uint32_t m_lastAnimationTick = ~std::uint32_t{0};
    /// ANI4+ANI5 renderer half: the shared GPU deformation pass (skin +
    /// morphs), created for deforming scenes in onSceneLoaded (reset on scene
    /// unload).
    std::optional<harmonia::scene::DeformPass> m_deformPass;
    bool m_animatedCaptureWarned = false; ///< one-time --output-without-sequence hint
};
} // namespace hyperion
#endif // DEMO_APPLICATION_HPP
