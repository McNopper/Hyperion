// Module tests: ShaderBindingTable spec compliance.
//
// The raygen SBT region must satisfy VUID-vkCmdTraceRaysKHR-size-04023: its size must equal
// the stride (always exactly 1 raygen shader), not alignUp(stride, baseAlignment). The
// validation layer tolerates an over-sized region; a bare driver does not.
// Use stride for both size and stride of the raygen VkStridedDeviceAddressRegionKHR.

#include <volk/volk.h>

#include <filesystem>
#include <gtest/gtest.h>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/renderer/Descriptors.hpp"
#include "harmonia/renderer/Pipeline.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"

namespace {
[[nodiscard]] std::filesystem::path shaderRoot() {
#ifdef HYPERION_SHADER_DIR
    return {HYPERION_SHADER_DIR};
#else
    return std::filesystem::path("build") / "shaders";
#endif
}

[[nodiscard]] Pipeline::ShaderPaths makeShaderPaths(const std::filesystem::path& root) {
    return Pipeline::ShaderPaths{
        .raygen = root / "raygen.spv",
        .closesthitTriangle = root / "closesthit.spv",
        .closesthitSphere = root / "closesthit.spv",
        .intersection = root / "intersection.spv",
        .miss = root / "miss.spv",
        .shadowMiss = root / "shadow_miss.spv",
    };
}
} // namespace

// Fixture that creates Descriptors + Pipeline + SBT once per test.
class SbtFixture : public RtFixture {
  protected:
    void SetUp() override {
        RtFixture::SetUp();

        const auto paths = makeShaderPaths(shaderRoot());
        if (!std::filesystem::exists(paths.raygen) || !std::filesystem::exists(paths.miss) ||
            !std::filesystem::exists(paths.closesthitTriangle) || !std::filesystem::exists(paths.intersection)) {
            GTEST_SKIP() << "Compiled shaders not found under " << shaderRoot().string();
        }

        auto desc = Descriptors::create(deviceCtx());
        ASSERT_TRUE(desc.has_value()) << "Descriptors::create failed: " << static_cast<int>(desc.error());
        m_descriptors = std::make_unique<Descriptors>(std::move(*desc));

        auto pipeline = Pipeline::create(deviceCtx(), *m_descriptors, paths, 2U);
        ASSERT_TRUE(pipeline.has_value()) << "Pipeline::create failed: " << static_cast<int>(pipeline.error());
        m_pipeline = std::make_unique<Pipeline>(std::move(*pipeline));

        auto sbt = ShaderBindingTable::create(deviceCtx(), *m_pipeline, physInfo().rtProps);
        ASSERT_TRUE(sbt.has_value()) << "ShaderBindingTable::create failed: " << static_cast<int>(sbt.error());
        m_sbt = std::make_unique<ShaderBindingTable>(std::move(*sbt));
    }

    std::unique_ptr<Descriptors> m_descriptors;
    std::unique_ptr<Pipeline> m_pipeline;
    std::unique_ptr<ShaderBindingTable> m_sbt;
};

// VUID-vkCmdTraceRaysKHR-size-04023: the raygen region size MUST equal stride
// (there is always exactly one raygen shader) — not alignUp(stride, baseAlignment).
TEST_F(SbtFixture, ShaderBindingTable_RaygenSizeEqualsStride) {
    const auto& raygen = m_sbt->raygenRegion();
    EXPECT_EQ(raygen.size, raygen.stride) << "Raygen region: size (" << raygen.size << ") != stride (" << raygen.stride
                                          << ") — violates VUID-vkCmdTraceRaysKHR-size-04023";
    EXPECT_GT(raygen.stride, VkDeviceSize{0}) << "Raygen stride must be non-zero";
    EXPECT_NE(raygen.deviceAddress, VkDeviceAddress{0}) << "Raygen device address must be non-zero";
}

// Miss and hit group SBT regions must have valid device addresses and strides.
TEST_F(SbtFixture, ShaderBindingTable_MissAndHitRegionsNonZero) {
    EXPECT_NE(m_sbt->missRegion().deviceAddress, VkDeviceAddress{0}) << "Miss region address";
    EXPECT_GT(m_sbt->missRegion().stride, VkDeviceSize{0}) << "Miss region stride";
    EXPECT_GT(m_sbt->missRegion().size, VkDeviceSize{0}) << "Miss region size";

    EXPECT_NE(m_sbt->hitRegion().deviceAddress, VkDeviceAddress{0}) << "Hit region address";
    EXPECT_GT(m_sbt->hitRegion().stride, VkDeviceSize{0}) << "Hit region stride";
    EXPECT_GT(m_sbt->hitRegion().size, VkDeviceSize{0}) << "Hit region size";
}

// Pipeline's RT pipeline handle must be non-null after creation.
TEST_F(SbtFixture, Pipeline_HandlesNonNull) {
    EXPECT_NE(m_pipeline->rtPipeline(), VK_NULL_HANDLE) << "RT pipeline handle";
}
