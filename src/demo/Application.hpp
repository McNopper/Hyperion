#pragma once

#include <volk/volk.h>

#include <SDL3/SDL.h>

#include <array>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "demo/presentation/Swapchain.hpp"
#include "demo/presentation/ToneMapper.hpp"
#include "demo/vulkan_init/Context.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/core/Image.hpp"
#include "hyperion/renderer/Camera.hpp"
#include "hyperion/renderer/Descriptors.hpp"
#include "hyperion/renderer/PathTracer.hpp"
#include "hyperion/renderer/Pipeline.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"
#include "hyperion/scene/IblProbe.hpp"
#include "hyperion/scene/Scene.hpp"

class Application {
  public:
    struct Config {
        std::string title = "Hyperion — Real-Time Path Tracer";
        uint32_t width = 1920;
        uint32_t height = 1080;
        uint32_t spp = 4;
        uint32_t maxDepth = 8;
        bool validation = false;
        bool sppExplicit = false; ///< true if --spp was given on the command line
        std::filesystem::path shaderDir = "";
        std::filesystem::path assetsDir = "assets";
        /// Scene definition file (.scene).  Path is resolved from the
        /// working directory.  Defaults to the classic Cornell box scene.
        std::filesystem::path sceneFile = "assets/cornell_classic.scene";
        /// If set, save this EXR after spp samples are accumulated, then exit.
        /// An empty path means interactive mode (default).
        std::filesystem::path outputFile;
    };

    [[nodiscard]] static std::expected<std::unique_ptr<Application>, int> create(Config config);

    Application() = default;
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&& other) noexcept;
    Application& operator=(Application&& other) noexcept;
    ~Application();

    int run();

  private:
    struct FrameResources {
        VkCommandBuffer traceCmd{};   ///< path trace recording
        VkCommandBuffer displayCmd{}; ///< tonemap recording (interactive only)
        VkSemaphore imageAvailable{};
        uint64_t completionValue{}; ///< highest timeline value signalled for this slot
    };

    void destroy() noexcept;

    /// Accumulates one path-traced sample into hdrTarget.
    /// Returns the timeline semaphore value that will be signalled when the
    /// buffer is ready for further processing (tonemap, EXR save, etc.).
    uint64_t renderFrame(Image& hdrTarget);
    void handleResize(uint32_t w, uint32_t h);
    void saveEXR(const std::filesystem::path& path);
    /// Tone-maps the HDR image (ACES SDR) and writes an 8-bit sRGB PNG.
    void savePNG(const std::filesystem::path& path);

    Config m_config{};
    SDL_Window* m_window{};
    Context m_context{};
    CommandPool m_commandPool{};
    Swapchain m_swapchain{};
    Descriptors m_descriptors{};
    Pipeline m_pipeline{};
    ShaderBindingTable m_sbt{};
    ToneMapper m_toneMapper{};
    PathTracer m_pathTracer{};
    Scene m_scene{};
    Camera m_camera{};
    Image m_hdrImage{};
    Image m_gNormal{};     ///< G-buffer world-space normal (R16G16B16A16_SFLOAT)
    Image m_gDepth{};      ///< G-buffer ray hit distance  (R32_SFLOAT)
    IblProbe m_iblProbe{}; ///< IBL equirectangular panorama (may be empty)
    std::array<FrameResources, 2> m_frames{};
    /// One binary semaphore per swapchain image: signalled by the display submit,
    /// consumed by vkQueuePresentKHR.  Indexed by swapchain imageIndex (not frame slot)
    /// to avoid reuse before the presentation engine has consumed the signal.
    std::vector<VkSemaphore> m_renderComplete;
    VkSemaphore m_timelineSemaphore{};
    uint64_t m_nextTimelineValue = 1;
    uint32_t m_currentFrame = 0;
    uint32_t m_frameIndex = 0;
    std::vector<VkImageLayout> m_swapchainLayouts;
    bool m_running = false;
};
