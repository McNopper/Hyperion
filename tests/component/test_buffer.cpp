// Component tests: Buffer lifecycle, upload, readback.
//
// These tests specifically target:
//   - Regression: Buffer::destroy() must NOT call vmaUnmapMemory on persistently-mapped
//     allocations (VMA_ALLOCATION_CREATE_MAPPED_BIT). The bug caused a VMA assertion in
//     Debug and silent memory corruption in Release.
//   - Host-mapped round-trip: upload bytes, read back via mappedData(), verify.
//   - Device-local staging upload: upload via staging path, copy back, verify data.
//   - Device address: buffers with SHADER_DEVICE_ADDRESS must return a non-zero address.

#include <array>
#include <cstring>

#include <gtest/gtest.h>
#include <volk/volk.h>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/core/Buffer.hpp"

namespace {
// Copy device-local buffer to a host-visible readback buffer and return the bytes.
[[nodiscard]] std::vector<uint8_t>
readbackBuffer(const DeviceContext& ctx, CommandPool& pool, const Buffer& src, VkDeviceSize size) {
    auto readback = Buffer::create(ctx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.readback");
    if (!readback || readback->mappedData() == nullptr) {
        return {};
    }
    auto cmd = pool.beginOneShot();
    if (!cmd) {
        return {};
    }
    const VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = size};
    vkCmdCopyBuffer(*cmd, src.handle(), readback->handle(), 1, &region);
    if (pool.endOneShot(*cmd) != VK_SUCCESS) {
        return {};
    }
    std::vector<uint8_t> result(static_cast<size_t>(size));
    std::memcpy(result.data(), readback->mappedData(), result.size());
    return result;
}
} // namespace

// Regression: Buffer::destroy() previously called vmaUnmapMemory() on allocations that
// were created with VMA_ALLOCATION_CREATE_MAPPED_BIT (persistent map). VMA asserts in
// Debug and corrupts memory in Release when you explicitly unmap a persistent allocation.
// Fix: removed the vmaUnmapMemory call — vmaDestroyBuffer handles it automatically.
TEST_F(VulkanFixture, Buffer_DestroyMappedBufferDoesNotCrash) {
    // Create and destroy several host-visible mapped buffers.
    // If the regression reappears the VMA abort() fires here in Debug.
    for (int i = 0; i < 8; ++i) {
        auto buf = Buffer::create(deviceCtx(), 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.mapped.destroy");
        ASSERT_TRUE(buf.has_value()) << "iter " << i << ": VkResult=" << static_cast<int>(buf.error());
        EXPECT_NE(buf->mappedData(), nullptr)
            << "iter " << i << ": host buffer must expose a persistent mapped pointer";
        EXPECT_NE(buf->handle(), VK_NULL_HANDLE);
        // ~Buffer() runs here — must not assert or crash
    }
}

// Upload a known byte pattern to a host-visible buffer, verify via mappedData().
TEST_F(VulkanFixture, Buffer_HostMappedRoundTrip) {
    constexpr size_t kCount = 64;
    std::array<uint8_t, kCount> pattern{};
    for (uint8_t i = 0; i < kCount; ++i) {
        pattern[i] = i;
    }

    auto buf = Buffer::create(deviceCtx(), kCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.host.roundtrip");
    ASSERT_TRUE(buf.has_value()) << static_cast<int>(buf.error());
    ASSERT_NE(buf->mappedData(), nullptr);

    buf->uploadData(pattern.data(), kCount);

    const auto* got = static_cast<const uint8_t*>(buf->mappedData());
    for (size_t i = 0; i < kCount; ++i) {
        EXPECT_EQ(got[i], pattern[i]) << "mismatch at byte " << i;
    }
}

// Upload known floats to a device-local buffer via the staging path, copy to a
// host-visible readback buffer, and verify the data survived the round-trip.
TEST_F(VulkanFixture, Buffer_DeviceLocalStagingUploadRoundTrip) {
    constexpr std::array<float, 16> kData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    constexpr VkDeviceSize kSize          = sizeof(kData);

    // TRANSFER_SRC_BIT so we can copy FROM it for readback.
    // Buffer::create always adds TRANSFER_DST_BIT, so staging upload is allowed.
    auto buf = Buffer::create(deviceCtx(), kSize,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, "test.device.staging");
    ASSERT_TRUE(buf.has_value()) << static_cast<int>(buf.error());
    EXPECT_EQ(buf->mappedData(), nullptr)
        << "device-local buffer should not have a host-accessible mapped pointer";

    buf->uploadData(kData.data(), kSize);

    const auto raw = readbackBuffer(deviceCtx(), commandPool(), *buf, kSize);
    ASSERT_EQ(raw.size(), static_cast<size_t>(kSize));

    std::array<float, 16> got{};
    std::memcpy(got.data(), raw.data(), kSize);
    for (size_t i = 0; i < kData.size(); ++i) {
        EXPECT_FLOAT_EQ(got[i], kData[i]) << "mismatch at element " << i;
    }
}

// A buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT must return a non-zero address.
TEST_F(VulkanFixture, Buffer_DeviceAddressNonZero) {
    auto buf = Buffer::create(deviceCtx(), 256,
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, "test.device.address");
    ASSERT_TRUE(buf.has_value()) << static_cast<int>(buf.error());
    EXPECT_NE(buf->deviceAddress(), VkDeviceAddress{0})
        << "buffer with SHADER_DEVICE_ADDRESS_BIT must have a non-zero device address";
}
