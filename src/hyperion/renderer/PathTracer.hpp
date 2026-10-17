#ifndef HYPERION_RENDERER_PATHTRACER_HPP
#define HYPERION_RENDERER_PATHTRACER_HPP

#include <volk/volk.h>

#include <cstdint>
#include <expected>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
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
           const Config& config);

    VkResult render(VkCommandBuffer cmd,
                    const hyperion::Scene& scene,
                    const harmonia::renderer::Camera& camera,
                    const harmonia::core::Image& hdrImage,
                    const harmonia::core::Image& gNormal,
                    const harmonia::core::Image& gDepth,
                    std::uint32_t frameIndex) noexcept;

    void setConfig(const Config& config) noexcept;
    void onResize(VkExtent2D newExtent) noexcept;

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
};
} // namespace hyperion
#endif // HYPERION_RENDERER_PATHTRACER_HPP
