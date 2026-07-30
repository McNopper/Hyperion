#include "hyperion/renderer/ShaderBindingTable.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "harmonia/core/Buffer.hpp"
#include "harmonia/renderer/Pipeline.hpp"

std::expected<ShaderBindingTable, VkResult>
ShaderBindingTable::create(const harmonia::DeviceContext& ctx,
                           const harmonia::Pipeline& pipeline,
                           const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProps) {
    constexpr std::uint32_t raygenCount = 1;
    constexpr std::uint32_t missCount = 2;
    constexpr std::uint32_t hitCount = 2;
    constexpr std::uint32_t groupCount = raygenCount + missCount + hitCount;

    const std::uint32_t handleSize = rtProps.shaderGroupHandleSize;
    const std::uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    const std::uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;
    const std::uint32_t stride = static_cast<std::uint32_t>(harmonia::bufferAlignUp(handleSize, handleAlignment));

    std::vector<std::byte> handles(static_cast<std::size_t>(groupCount) * handleSize);
    if (const VkResult result = vkGetRayTracingShaderGroupHandlesKHR(
            ctx.device, pipeline.rtPipeline(), 0, groupCount, handles.size(), handles.data());
        result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    const VkDeviceSize raygenOffset = 0;
    const VkDeviceSize raygenSize = harmonia::bufferAlignUp(static_cast<VkDeviceSize>(stride) * raygenCount, baseAlignment);
    const VkDeviceSize missOffset = harmonia::bufferAlignUp(raygenOffset + raygenSize, baseAlignment);
    const VkDeviceSize missSize = harmonia::bufferAlignUp(static_cast<VkDeviceSize>(stride) * missCount, baseAlignment);
    const VkDeviceSize hitOffset = harmonia::bufferAlignUp(missOffset + missSize, baseAlignment);
    const VkDeviceSize hitSize = harmonia::bufferAlignUp(static_cast<VkDeviceSize>(stride) * hitCount, baseAlignment);
    const VkDeviceSize totalSize = hitOffset + hitSize;

    std::vector<std::byte> sbtBytes(static_cast<std::size_t>(totalSize), std::byte{0});
    auto copyHandle = [&](std::uint32_t groupIndex, VkDeviceSize dstOffset) {
        std::memcpy(sbtBytes.data() + static_cast<std::size_t>(dstOffset),
                    handles.data() + static_cast<std::size_t>(groupIndex) * handleSize,
                    handleSize);
    };
    copyHandle(0, raygenOffset);
    for (std::uint32_t i = 0; i < missCount; ++i) {
        copyHandle(raygenCount + i, missOffset + static_cast<VkDeviceSize>(stride) * i);
    }
    for (std::uint32_t i = 0; i < hitCount; ++i) {
        copyHandle(raygenCount + missCount + i, hitOffset + static_cast<VkDeviceSize>(stride) * i);
    }

    auto buffer =
        harmonia::Buffer::create(ctx,
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
    return sbt;
}
