#pragma once

#include <volk/volk.h>

#include <filesystem>

#include "harmonia/app/App.hpp"
#include "harmonia/app/IRenderer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/Scene.hpp"

/// Hyperion demo: the path tracer injected into the shared harmonia::App host.
///
/// The host owns window/context/swapchain/HDR image/tonemap/present; this
/// class owns only what is path-tracer specific — RT pipeline + SBT,
/// G-buffers, the Scene, the camera and the PathTracer itself.  Per the host
/// contract, record() produces a linear image in the scene-referred working
/// color space and leaves it in VK_IMAGE_LAYOUT_GENERAL.
class Application final : public harmonia::App, public harmonia::IRenderer {
  public:
    struct DemoConfig {
        uint32_t spp = 4;
        uint32_t maxDepth = 8;
        bool sppExplicit = false; ///< true if --spp was given on the command line
        std::filesystem::path shaderDir;
    };

    int run(Config config, DemoConfig demoConfig);

    // harmonia::IRenderer
    void record(VkCommandBuffer cmd, const harmonia::RenderTarget& target) noexcept override;
    void onResize(VkExtent2D extent) noexcept override;
    [[nodiscard]] VkPipelineStageFlags2 outputStageMask() const noexcept override {
        return VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    }
    [[nodiscard]] const char* name() const noexcept override { return "Hyperion PathTracer"; }

  protected:
    // harmonia::App hooks
    [[nodiscard]] harmonia::IRenderer& renderer() noexcept override { return *this; }
    [[nodiscard]] ISceneBuilder& sceneBuilder() noexcept override { return m_scene; }
    [[nodiscard]] bool onInitialize() override;
    [[nodiscard]] bool onSceneLoaded(const SceneLoader::SceneConfig& sceneConfig) override;
    void onSceneUnload() override;
    bool onEvent(const SDL_Event& event) override;
    void onResized(VkExtent2D extent) override;
    [[nodiscard]] uint32_t offscreenFrameCount() const noexcept override { return m_demoConfig.spp; }

  private:
    [[nodiscard]] bool createGBuffers(VkExtent2D extent);

    DemoConfig m_demoConfig{};
    std::filesystem::path m_shaderDir;
    Pipeline m_pipeline{};
    ShaderBindingTable m_sbt{};
    PathTracer m_pathTracer{};
    Scene m_scene{};
    Camera m_camera{};
    Image m_gNormal{}; ///< G-buffer world-space normal (R16G16B16A16_SFLOAT)
    Image m_gDepth{};  ///< G-buffer ray hit distance  (R32_SFLOAT)
    bool m_targetsFirstUse = true; ///< HDR/G-buffer images need UNDEFINED→GENERAL
};
