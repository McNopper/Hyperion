#ifndef HYPERION_RENDERER_SHADERBINDINGTABLE_HPP
#define HYPERION_RENDERER_SHADERBINDINGTABLE_HPP

#include <volk/volk.h>

#include <expected>

#include "harmonia/DeviceContext.hpp"
#include "harmonia/core/Buffer.hpp"

namespace harmonia {
class Pipeline;
}

class ShaderBindingTable {
  public:
    [[nodiscard]] static std::expected<ShaderBindingTable, VkResult>
    create(const harmonia::DeviceContext& ctx,
           const harmonia::Pipeline& pipeline,
           const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProps);

    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& raygenRegion() const noexcept { return m_raygen; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& missRegion() const noexcept { return m_miss; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& hitRegion() const noexcept { return m_hit; }

  private:
    harmonia::Buffer m_sbtBuffer{};
    VkStridedDeviceAddressRegionKHR m_raygen{};
    VkStridedDeviceAddressRegionKHR m_miss{};
    VkStridedDeviceAddressRegionKHR m_hit{};
};
#endif // HYPERION_RENDERER_SHADERBINDINGTABLE_HPP
