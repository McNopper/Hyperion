#ifndef DEMO_APPLICATION_HPP
#define DEMO_APPLICATION_HPP

#include <volk/volk.h>

#include <cstdint>
#include <filesystem>

#include "harmonia/app/App.hpp"
#include "harmonia/app/IRenderer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/renderer/PathTracer.hpp"
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
class Application final : public harmonia::app::App, public harmonia::app::IRenderer {
  public:
    struct DemoConfig {
        std::uint32_t spp = 4;
        std::uint32_t maxDepth = 8;
        bool sppExplicit = false; ///< true if --spp was given on the command line
        bool maxDepthExplicit = false; ///< true if --depth was given on the command line
        std::filesystem::path shaderDir;
    };

    int run(Config&& config, DemoConfig&& demoConfig);

    // harmonia::app::IRenderer
    void record(VkCommandBuffer cmd, const harmonia::app::RenderTarget& target) noexcept override;
    void onResize(VkExtent2D extent) noexcept override;
    [[nodiscard]] VkPipelineStageFlags2 outputStageMask() const noexcept override {
        return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }
    [[nodiscard]] VkImageView gNormalView() const noexcept override { return m_gNormal.view(); }
    [[nodiscard]] VkImageView gDepthView() const noexcept override { return m_gDepth.view(); }
    [[nodiscard]] const char* name() const noexcept override { return "Hyperion hyperion::PathTracer"; }

  protected:
    // harmonia::app::App hooks
    [[nodiscard]] harmonia::app::IRenderer& renderer() noexcept override { return *this; }
    [[nodiscard]] harmonia::scene::ISceneBuilder& sceneBuilder() noexcept override { return m_scene; }
    [[nodiscard]] bool onInitialize() override;
    [[nodiscard]] bool onSceneLoaded(const harmonia::scene::SceneLoader::SceneConfig& sceneConfig) override;
    void onSceneUnload() override;
    bool onEvent(const SDL_Event& event) override;
    [[nodiscard]] std::uint32_t offscreenFrameCount() const noexcept override { return m_demoConfig.spp; }

  private:
    [[nodiscard]] bool createGBuffers(VkExtent2D extent);
    void applySceneOverrides(const harmonia::scene::SceneLoader::SceneConfig& config);
    void buildCamera(const harmonia::scene::SceneLoader::SceneConfig& config);
    [[nodiscard]] VkResult setupSceneDescriptors();
    void transitionTargetsOnFirstUse(VkCommandBuffer cmd) noexcept;

    DemoConfig m_demoConfig{};
    std::filesystem::path m_shaderDir;
    harmonia::renderer::Pipeline m_pipeline{};
    hyperion::ShaderBindingTable m_sbt{};
    hyperion::PathTracer m_pathTracer{};
    hyperion::Scene m_scene{};
    harmonia::renderer::Camera m_camera{};
    harmonia::core::Image m_gNormal{};           ///< G-buffer world-space normal (R16G16B16A16_SFLOAT)
    harmonia::core::Image m_gDepth{};            ///< G-buffer ray hit distance  (R32_SFLOAT)
    bool m_targetsFirstUse = true;         ///< HDR/G-buffer images need UNDEFINED→GENERAL
    bool m_positionFetchSupported = false; ///< VK_KHR_ray_tracing_position_fetch is active.
};
} // namespace hyperion
#endif // DEMO_APPLICATION_HPP
