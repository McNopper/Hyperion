#pragma once

#include <volk/volk.h>

#include <expected>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Image.hpp"
#include "harmonia/renderer/Camera.hpp"

class Descriptors;
class Pipeline;
class Scene;
class ShaderBindingTable;

class PathTracer {
  public:
    struct Config {
        uint32_t samplesPerPixel = 4;
        uint32_t maxDepth = 8;
        float envLuminance = 1.0f;
        uint32_t outputColorSpace = 0;    ///< OutputColorSpace enum value; 0 = eHDR10 (see OutputColorSpace.hpp)
        uint32_t hasEnvMap = 0;           ///< 1 = IBL env map bound at set1/binding6
        uint32_t envImportanceWidth = 0;  ///< CDF grid width; 0 = importance sampling disabled
        uint32_t envImportanceHeight = 0; ///< CDF grid height
        uint32_t tonemapper = 0;          ///< Tonemapper enum value; 0 = eACES (SDR/P3 only)
        /// ColorSpace::WorkingColorSpace value of the scene-referred working
        /// space (0 = linear Rec.2020, 1 = linear Rec.709); forwarded to the
        /// tonemap push constant.
        uint32_t workingColorSpace = 0;
        bool serEnabled = false;      ///< Enable VK_EXT_ray_tracing_invocation_reorder when supported.
        bool indirectRt2Enabled = false; ///< Use vkCmdTraceRaysIndirect2KHR when supported.
    };

    [[nodiscard]] static std::expected<PathTracer, VkResult> create(const DeviceContext& ctx,
                                                                    VkExtent2D renderExtent,
                                                                    const Pipeline& pipeline,
                                                                    const ShaderBindingTable& sbt,
                                                                    const Descriptors& descriptors,
                                                                    const Config& config);

    [[nodiscard]] static std::expected<PathTracer, VkResult> create(const DeviceContext& ctx,
                                                                    VkExtent2D renderExtent,
                                                                    const Pipeline& pipeline,
                                                                    const ShaderBindingTable& sbt,
                                                                    const Descriptors& descriptors);

    VkResult render(VkCommandBuffer cmd,
                    const Scene& scene,
                    const Camera& camera,
                    const Image& hdrImage,
                    const Image& gNormal,
                    const Image& gDepth,
                    uint32_t frameIndex) noexcept;

    void setConfig(const Config& config) noexcept;
    void onResize(VkExtent2D newExtent) noexcept;

  private:
    void updateIndirectBuffer() noexcept;
    const DeviceContext* m_ctx{};
    VkPipeline m_rtPipeline{VK_NULL_HANDLE};
    VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
    VkDescriptorSet m_sceneSet{VK_NULL_HANDLE};
    VkExtent2D m_extent{};
    Config m_config;
    Buffer m_cameraBuffer{};
    Buffer m_indirectDispatchBuffer{}; ///< GPU buffer for VkTraceRaysIndirectCommand2KHR (B2).
    VkStridedDeviceAddressRegionKHR m_raygen{};
    VkStridedDeviceAddressRegionKHR m_miss{};
    VkStridedDeviceAddressRegionKHR m_hit{};
    VkStridedDeviceAddressRegionKHR m_callable{};
};
