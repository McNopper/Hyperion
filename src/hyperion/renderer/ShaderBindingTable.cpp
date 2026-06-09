#include "hyperion/renderer/ShaderBindingTable.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

#include "harmonia/renderer/Pipeline.hpp"

namespace {
[[nodiscard]] constexpr VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) noexcept {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}
} // namespace

std::expected<ShaderBindingTable, VkResult>
ShaderBindingTable::create(const DeviceContext& ctx,
                           const Pipeline& pipeline,
                           VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps) {
    constexpr uint32_t groupCount = 5;
    constexpr uint32_t raygenCount = 1;
    constexpr uint32_t missCount = 2;
    constexpr uint32_t hitCount = 2;

    const uint32_t handleSize = rtProps.shaderGroupHandleSize;
    const uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    const uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;
    const uint32_t stride = static_cast<uint32_t>(alignUp(handleSize, handleAlignment));

    std::vector<std::byte> handles(static_cast<size_t>(groupCount) * handleSize);
    if (const VkResult result = vkGetRayTracingShaderGroupHandlesKHR(
            ctx.device, pipeline.rtPipeline(), 0, groupCount, handles.size(), handles.data());
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    const VkDeviceSize raygenOffset = 0;
    const VkDeviceSize raygenSize = alignUp(stride * raygenCount, baseAlignment);
    const VkDeviceSize missOffset = alignUp(raygenOffset + raygenSize, baseAlignment);
    const VkDeviceSize missSize = alignUp(stride * missCount, baseAlignment);
    const VkDeviceSize hitOffset = alignUp(missOffset + missSize, baseAlignment);
    const VkDeviceSize hitSize = alignUp(stride * hitCount, baseAlignment);
    const VkDeviceSize totalSize = hitOffset + hitSize;

    std::vector<std::byte> sbtBytes(static_cast<size_t>(totalSize), std::byte{0});
    auto copyHandle = [&](uint32_t groupIndex, VkDeviceSize dstOffset) {
        std::memcpy(sbtBytes.data() + static_cast<size_t>(dstOffset),
                    handles.data() + static_cast<size_t>(groupIndex) * handleSize,
                    handleSize);
    };
    copyHandle(0, raygenOffset);
    copyHandle(1, missOffset + static_cast<VkDeviceSize>(stride) * 0);
    copyHandle(2, missOffset + static_cast<VkDeviceSize>(stride) * 1);
    copyHandle(3, hitOffset + static_cast<VkDeviceSize>(stride) * 0);
    copyHandle(4, hitOffset + static_cast<VkDeviceSize>(stride) * 1);

    auto buffer =
        Buffer::create(ctx,
                       totalSize,
                       VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                       "hyperion.sbt");
    if (!buffer) {
        return std::unexpected(buffer.error());
    }
    buffer->uploadData(sbtBytes.data(), sbtBytes.size(), 0);

    ShaderBindingTable sbt;
    sbt.m_sbtBuffer = std::move(*buffer);

    const VkDeviceAddress baseAddress = sbt.m_sbtBuffer.deviceAddress();
    // For raygen, the spec requires size == stride (single entry, always one raygen shader).
    sbt.m_raygen = VkStridedDeviceAddressRegionKHR{baseAddress + raygenOffset, stride, stride};
    sbt.m_miss = VkStridedDeviceAddressRegionKHR{baseAddress + missOffset, stride, missSize};
    sbt.m_hit = VkStridedDeviceAddressRegionKHR{baseAddress + hitOffset, stride, hitSize};
    sbt.m_callable = VkStridedDeviceAddressRegionKHR{};
    return sbt;
}
