#pragma once

#include <volk/volk.h>

#include <expected>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/Buffer.hpp"

class Pipeline;

class ShaderBindingTable {
  public:
    [[nodiscard]] static std::expected<ShaderBindingTable, VkResult>
    create(const DeviceContext& ctx, const Pipeline& pipeline, VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps);

    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& raygenRegion() const noexcept { return m_raygen; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& missRegion() const noexcept { return m_miss; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& hitRegion() const noexcept { return m_hit; }
    [[nodiscard]] const VkStridedDeviceAddressRegionKHR& callableRegion() const noexcept { return m_callable; }

  private:
    Buffer m_sbtBuffer{};
    VkStridedDeviceAddressRegionKHR m_raygen{};
    VkStridedDeviceAddressRegionKHR m_miss{};
    VkStridedDeviceAddressRegionKHR m_hit{};
    VkStridedDeviceAddressRegionKHR m_callable{};
};
