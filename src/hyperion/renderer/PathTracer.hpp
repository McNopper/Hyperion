#ifndef HYPERION_RENDERER_PATHTRACER_HPP
#define HYPERION_RENDERER_PATHTRACER_HPP

#include <volk/volk.h>

#include <cstdint>
#include <expected>
#include <filesystem>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/pipeline/PassContext.hpp"
#include "harmonia/pipeline/SmsPass.hpp"
#include "harmonia/renderer/Camera.hpp"
#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/utils/ColorSpace.hpp"
#include "harmonia/utils/OutputColorSpace.hpp"

namespace hyperion {
class Descriptors;
}
namespace harmonia::renderer {
class Pipeline;
}
namespace hyperion {
class Scene;
class ShaderBindingTable;
} // namespace hyperion

namespace hyperion {
class PathTracer {
  public:
    struct Config {
        std::uint32_t samplesPerPixel = 4;
        std::uint32_t maxDepth = 8;
        std::uint32_t rngSeed = harmonia::pipeline::kDefaultRngSeed;
        std::uint32_t smsPartitionResolution = 16; ///< Hong 3x3 trust regions; 1 is the base-SMS limit.
        bool smsPriorEnabled = true; ///< Per-frame pilot union; N=1 is already uniform and needs no prior.
        /// GI-SMS #2: off selects the ordinary-PT reference — no SMS work is collected
        /// (shaders see partition resolution 0) and no priors are built. The SMS queue
        /// machinery stays in place and completes immediately with zero work.
        bool smsEnabled = true;
        float envLuminance = 1.0f;
        harmonia::utils::OutputColorSpace outputColorSpace = harmonia::utils::OutputColorSpace::eHDR10;
        std::uint32_t hasEnvMap = 0;           ///< 1 = IBL env map bound at set1/binding6
        std::uint32_t envImportanceWidth = 0;  ///< CDF grid width; 0 = importance sampling disabled
        std::uint32_t envImportanceHeight = 0; ///< CDF grid height
        harmonia::Tonemapper tonemapper = harmonia::Tonemapper::eACES;
        harmonia::utils::ColorSpace::WorkingColorSpace workingColorSpace =
            harmonia::utils::ColorSpace::WorkingColorSpace::LinRec2020;
        bool serEnabled = false; ///< Enable VK_EXT_ray_tracing_invocation_reorder when supported.
    };

    [[nodiscard]] static std::expected<hyperion::PathTracer, VkResult>
    create(const harmonia::DeviceContext& ctx,
           VkExtent2D renderExtent,
           const harmonia::renderer::Pipeline& pipeline,
           const hyperion::ShaderBindingTable& sbt,
           harmonia::renderer::Descriptors& descriptors,
           const Config& config,
           const std::filesystem::path& shaderDirectory = {});

    /// Records bounded initial work. Output is usable only after a waited
    /// frameReady() succeeds; resume with recordContinuation while pending.
    VkResult render(VkCommandBuffer cmd,
                    const hyperion::Scene& scene,
                    const harmonia::renderer::Camera& camera,
                    const harmonia::core::Image& hdrImage,
                    const harmonia::core::Image& gNormal,
                    const harmonia::core::Image& gDepth,
                    std::uint32_t frameIndex) noexcept;

    /// Retire the previous frame before changing sampling capacity/resizing.
    void setConfig(const Config& config) noexcept;
    void onResize(VkExtent2D newExtent) noexcept;
    /// Queried only after the previous submission has retired. Ready includes
    /// full-slot validation AND HDR resolve; no pending work enters accumulation.
    [[nodiscard]] std::expected<bool, VkResult> frameReady() const noexcept;
    [[nodiscard]] VkResult recordContinuation(VkCommandBuffer cmd) noexcept;

  private:
    void writeFrameDescriptors(VkCommandBuffer cmd,
                               const hyperion::Scene& scene,
                               const harmonia::renderer::Camera& camera,
                               std::uint32_t frameIndex,
                               const harmonia::core::Image& hdrImage,
                               const harmonia::core::Image& gNormal,
                               const harmonia::core::Image& gDepth) noexcept;
    void pushFrameConstants(VkCommandBuffer cmd, const hyperion::Scene& scene, std::uint32_t frameIndex) noexcept;
    void dispatchRays(VkCommandBuffer cmd) noexcept;

    void updateIndirectBuffer() noexcept;
    [[nodiscard]] VkResult resetSmsWork();
    const harmonia::DeviceContext* m_ctx{};
    VkPipeline m_rtPipeline{VK_NULL_HANDLE};
    VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
    /// MOD1: descriptor buffer — scene set bound via Descriptors::bindSceneSet().
    harmonia::renderer::Descriptors* m_descriptors{nullptr};
    VkExtent2D m_extent{};
    Config m_config;
    harmonia::core::Buffer m_cameraBuffer{};
    harmonia::core::Buffer m_indirectDispatchBuffer{}; ///< GPU buffer for VkTraceRaysIndirectCommand2KHR (B2).
    VkStridedDeviceAddressRegionKHR m_raygen{};
    VkStridedDeviceAddressRegionKHR m_miss{};
    VkStridedDeviceAddressRegionKHR m_hit{};
    VkStridedDeviceAddressRegionKHR m_callable{};
    std::filesystem::path m_shaderDirectory;
    harmonia::pipeline::SmsPass m_sms;
    harmonia::core::Buffer m_priorBuffer;
    bool m_priorActive = false;
    VkResult m_frameError = VK_SUCCESS;
    bool m_frameRecorded = false;
};
} // namespace hyperion
#endif // HYPERION_RENDERER_PATHTRACER_HPP
