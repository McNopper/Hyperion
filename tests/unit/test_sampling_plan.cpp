#include <cstdint>
#include <gtest/gtest.h>

#include "hyperion/renderer/SamplingPlan.hpp"

TEST(SamplingPlan, CaptureUsesExactlyRequestedSamples) {
    for (std::uint32_t requested : {1U, 4U, 64U, 256U}) {
        const auto plan = hyperion::samplingPlan(requested, true);
        EXPECT_EQ(plan.frameCount, requested);
        EXPECT_EQ(plan.samplesPerDispatch, 1U);
        EXPECT_EQ(static_cast<std::uint64_t>(plan.frameCount) * plan.samplesPerDispatch, requested);
    }
}

TEST(SamplingPlan, InteractiveBatchingDoesNotDuplicateHistory) {
    const auto plan = hyperion::samplingPlan(64U, false);
    EXPECT_EQ(plan.frameCount, 1U);
    EXPECT_EQ(plan.samplesPerDispatch, 64U);
}

TEST(SamplingPlan, ZeroRequestClampsToOneSample) {
    for (bool capture : {false, true}) {
        const auto plan = hyperion::samplingPlan(0U, capture);
        EXPECT_EQ(plan.frameCount, 1U);
        EXPECT_EQ(plan.samplesPerDispatch, 1U);
    }
}
