// Module tests: ShaderBindingTable spec compliance.
//
// Regression: VUID-vkCmdTraceRaysKHR-size-04023 requires the raygen SBT region
// size to equal stride (always exactly 1 raygen shader). We previously computed
// size = alignUp(stride, baseAlignment) = 64 while stride = 32, violating the spec.
// The validation layer silently tolerated this; the bare driver crashed.
//
// Fix: use stride for both size and stride of the raygen VkStridedDeviceAddressRegionKHR.

#include <filesystem>

#include <gtest/gtest.h>
#include <volk/volk.h>

#include "fixtures/VulkanTestFixture.hpp"
#include "hyperion/renderer/Descriptors.hpp"
#include "hyperion/renderer/Pipeline.hpp"
#include "hyperion/renderer/ShaderBindingTable.hpp"

namespace {
[[nodiscard]] std::filesystem::path shaderRoot() {
#ifdef HYPERION_SHADER_DIR
    return std::filesystem::path(HYPERION_SHADER_DIR);
#else
    return std::filesystem::path("build") / "shaders";
#endif
}

[[nodiscard]] Pipeline::ShaderPaths makeShaderPaths(const std::filesystem::path& root) {
    return Pipeline::ShaderPaths{
        .raygen              = root / "raygen.spv",
        .closesthitTriangle  = root / "closesthit.spv",
        .closesthitSphere    = root / "closesthit.spv",
        .intersection        = root / "intersection.spv",
        .miss                = root / "miss.spv",
        .shadowMiss          = root / "shadow_miss.spv",
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
            !std::filesystem::exists(paths.closesthitTriangle) ||
            !std::filesystem::exists(paths.intersection)) {
            GTEST_SKIP() << "Compiled shaders not found under " << shaderRoot().string();
        }

        auto desc = Descriptors::create(deviceCtx());
        ASSERT_TRUE(desc.has_value()) << "Descriptors::create failed: " << static_cast<int>(desc.error());
        m_descriptors = std::make_unique<Descriptors>(std::move(*desc));

        auto pipeline = Pipeline::create(deviceCtx(), *m_descriptors, paths, 2U);
        ASSERT_TRUE(pipeline.has_value())
            << "Pipeline::create failed: " << static_cast<int>(pipeline.error());
        m_pipeline = std::make_unique<Pipeline>(std::move(*pipeline));

        auto sbt = ShaderBindingTable::create(deviceCtx(), *m_pipeline, physInfo().rtProps);
        ASSERT_TRUE(sbt.has_value()) << "ShaderBindingTable::create failed: " << static_cast<int>(sbt.error());
        m_sbt = std::make_unique<ShaderBindingTable>(std::move(*sbt));
    }

    std::unique_ptr<Descriptors>         m_descriptors;
    std::unique_ptr<Pipeline>            m_pipeline;
    std::unique_ptr<ShaderBindingTable>  m_sbt;
};

// Regression: VUID-vkCmdTraceRaysKHR-size-04023
// Raygen region size MUST equal stride (there is always exactly one raygen shader).
// Previous bug: size = alignUp(stride * 1, baseAlignment) = 64 (stride = 32).
TEST_F(SbtFixture, ShaderBindingTable_RaygenSizeEqualsStride) {
    const auto& raygen = m_sbt->raygenRegion();
    EXPECT_EQ(raygen.size, raygen.stride)
        << "Raygen region: size (" << raygen.size << ") != stride (" << raygen.stride
        << ") — violates VUID-vkCmdTraceRaysKHR-size-04023";
    EXPECT_GT(raygen.stride, VkDeviceSize{0}) << "Raygen stride must be non-zero";
    EXPECT_NE(raygen.deviceAddress, VkDeviceAddress{0}) << "Raygen device address must be non-zero";
}

// Miss and hit group SBT regions must have valid device addresses and strides.
TEST_F(SbtFixture, ShaderBindingTable_MissAndHitRegionsNonZero) {
    EXPECT_NE(m_sbt->missRegion().deviceAddress, VkDeviceAddress{0}) << "Miss region address";
    EXPECT_GT(m_sbt->missRegion().stride, VkDeviceSize{0})           << "Miss region stride";
    EXPECT_GT(m_sbt->missRegion().size, VkDeviceSize{0})             << "Miss region size";

    EXPECT_NE(m_sbt->hitRegion().deviceAddress, VkDeviceAddress{0}) << "Hit region address";
    EXPECT_GT(m_sbt->hitRegion().stride, VkDeviceSize{0})           << "Hit region stride";
    EXPECT_GT(m_sbt->hitRegion().size, VkDeviceSize{0})             << "Hit region size";
}

// Pipeline's RT pipeline handle must be non-null after creation.
TEST_F(SbtFixture, Pipeline_HandlesNonNull) {
    EXPECT_NE(m_pipeline->rtPipeline(), VK_NULL_HANDLE) << "RT pipeline handle";
}
