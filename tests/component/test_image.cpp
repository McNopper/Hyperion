// Component tests: Image creation, layout transition, and pixel readback.
//
// These tests verify:
//   - Image creation/destruction does not leak or crash.
//   - Layout transitions record correctly (UNDEFINED → TRANSFER_DST → TRANSFER_SRC).
//   - Image data written via vkCmdClearColorImage survives a GPU→CPU copy.

#include <cstdint>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <volk/volk.h>

#include "fixtures/VulkanTestFixture.hpp"
#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/Image.hpp"

TEST_F(VulkanFixture, Image_CreateAndDestroyR32G32B32A32) {
    constexpr VkExtent2D kExtent{16U, 16U};

    auto img = Image::create(deviceCtx(), kExtent, VK_FORMAT_R32G32B32A32_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT, "test.image.create");
    ASSERT_TRUE(img.has_value()) << "VkResult=" << static_cast<int>(img.error());
    EXPECT_NE(img->handle(), VK_NULL_HANDLE);
    EXPECT_NE(img->view(), VK_NULL_HANDLE);
    EXPECT_EQ(img->extent().width, kExtent.width);
    EXPECT_EQ(img->extent().height, kExtent.height);
    EXPECT_EQ(img->format(), VK_FORMAT_R32G32B32A32_SFLOAT);
    EXPECT_TRUE(img->isValid());
    // ~Image() must not crash or leak
}

// Transition image UNDEFINED→TRANSFER_DST, clear with a known color,
// transition to TRANSFER_SRC, copy to a host buffer, and verify all pixels.
TEST_F(VulkanFixture, Image_TransitionClearAndReadback) {
    constexpr VkExtent2D kExtent{8U, 8U};
    constexpr uint32_t   kPixelCount = kExtent.width * kExtent.height;
    constexpr VkDeviceSize kReadbackBytes = kPixelCount * sizeof(glm::vec4);

    // Expected clear value
    constexpr VkClearColorValue kClear{.float32 = {0.5F, 0.25F, 0.125F, 1.0F}};

    auto img = Image::create(deviceCtx(), kExtent, VK_FORMAT_R32G32B32A32_SFLOAT,
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT, "test.image.clear");
    ASSERT_TRUE(img.has_value()) << static_cast<int>(img.error());

    auto readback = Buffer::create(deviceCtx(), kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "test.image.readback");
    ASSERT_TRUE(readback.has_value()) << static_cast<int>(readback.error());
    ASSERT_NE(readback->mappedData(), nullptr);

    auto cmd = commandPool().beginOneShot();
    ASSERT_TRUE(cmd.has_value()) << static_cast<int>(cmd.error());

    // UNDEFINED → TRANSFER_DST for the clear
    img->transition(*cmd,
                    VK_IMAGE_LAYOUT_UNDEFINED,      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COPY_BIT,   VK_ACCESS_2_TRANSFER_WRITE_BIT);

    const VkImageSubresourceRange fullRange{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 0, .levelCount = 1,
        .baseArrayLayer = 0, .layerCount = 1,
    };
    vkCmdClearColorImage(*cmd, img->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &kClear, 1, &fullRange);

    // TRANSFER_DST → TRANSFER_SRC for copy-out
    img->transition(*cmd,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

    const VkBufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset       = {0, 0, 0},
        .imageExtent       = {kExtent.width, kExtent.height, 1},
    };
    vkCmdCopyImageToBuffer(*cmd, img->handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback->handle(), 1, &region);

    ASSERT_EQ(commandPool().endOneShot(*cmd), VK_SUCCESS);

    const auto* pixels = static_cast<const glm::vec4*>(readback->mappedData());
    for (uint32_t i = 0; i < kPixelCount; ++i) {
        EXPECT_NEAR(pixels[i].r, kClear.float32[0], 1e-5F) << "pixel " << i << " R";
        EXPECT_NEAR(pixels[i].g, kClear.float32[1], 1e-5F) << "pixel " << i << " G";
        EXPECT_NEAR(pixels[i].b, kClear.float32[2], 1e-5F) << "pixel " << i << " B";
        EXPECT_NEAR(pixels[i].a, kClear.float32[3], 1e-5F) << "pixel " << i << " A";
    }
}
