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
#include "harmonia/utils/ColorSpace.hpp"
#include "harmonia/utils/OutputColorSpace.hpp"

namespace harmonia { class Descriptors; }
namespace harmonia { class Pipeline; }
class Scene;
class ShaderBindingTable;

class PathTracer {
  public:
    struct Config {
        std::uint32_t samplesPerPixel = 4;
        std::uint32_t maxDepth = 8;
        float envLuminance = 1.0f;
        harmonia::OutputColorSpace outputColorSpace = harmonia::OutputColorSpace::eHDR10;
        std::uint32_t hasEnvMap = 0;           ///< 1 = IBL env map bound at set1/binding6
        std::uint32_t envImportanceWidth = 0;  ///< CDF grid width; 0 = importance sampling disabled
        std::uint32_t envImportanceHeight = 0; ///< CDF grid height
        harmonia::Tonemapper tonemapper = harmonia::Tonemapper::eACES;
        harmonia::ColorSpace::WorkingColorSpace workingColorSpace = harmonia::ColorSpace::WorkingColorSpace::LinRec2020;
        bool serEnabled = false;         ///< Enable VK_EXT_ray_tracing_invocation_reorder when supported.
        bool indirectRt2Enabled = false; ///< Use vkCmdTraceRaysIndirect2KHR when supported.
    };

    [[nodiscard]] static std::expected<PathTracer, VkResult> create(const harmonia::DeviceContext& ctx,
                                                                    VkExtent2D renderExtent,
                                                                    const harmonia::Pipeline& pipeline,
                                                                    const ShaderBindingTable& sbt,
                                                                    const harmonia::Descriptors& descriptors,
                                                                    const Config& config);

    VkResult render(VkCommandBuffer cmd,
                    const Scene& scene,
                    const harmonia::Camera& camera,
                    const harmonia::Image& hdrImage,
                    const harmonia::Image& gNormal,
                    const harmonia::Image& gDepth,
                    std::uint32_t frameIndex) noexcept;

    void setConfig(const Config& config) noexcept;
    void onResize(VkExtent2D newExtent) noexcept;

  private:
    void writeFrameDescriptors(VkCommandBuffer cmd,
                               const Scene& scene,
                               const harmonia::Camera& camera,
                               std::uint32_t frameIndex,
                               const harmonia::Image& hdrImage,
                               const harmonia::Image& gNormal,
                               const harmonia::Image& gDepth) noexcept;
    void pushFrameConstants(VkCommandBuffer cmd, const Scene& scene, std::uint32_t frameIndex) noexcept;
    void dispatchRays(VkCommandBuffer cmd) noexcept;
    void dispatchDirect(VkCommandBuffer cmd) noexcept;

    void updateIndirectBuffer() noexcept;
    const harmonia::DeviceContext* m_ctx{};
    VkPipeline m_rtPipeline{VK_NULL_HANDLE};
    VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
    VkDescriptorSet m_sceneSet{VK_NULL_HANDLE};
    VkExtent2D m_extent{};
    Config m_config;
    harmonia::Buffer m_cameraBuffer{};
    harmonia::Buffer m_indirectDispatchBuffer{}; ///< GPU buffer for VkTraceRaysIndirectCommand2KHR (B2).
    VkStridedDeviceAddressRegionKHR m_raygen{};
    VkStridedDeviceAddressRegionKHR m_miss{};
    VkStridedDeviceAddressRegionKHR m_hit{};
    VkStridedDeviceAddressRegionKHR m_callable{};
};
#endif // HYPERION_RENDERER_PATHTRACER_HPP
