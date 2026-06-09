
#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <array>
#include <gtest/gtest.h>

#include "harmonia/utils/ColorSpace.hpp"

namespace {
constexpr float kEpsilon = 1.0e-5F;
}

TEST(ColorSpace, SrgbRoundTripsWithinTolerance) {
    const std::array values{
        glm::vec3(0.0F),
        glm::vec3(0.18F),
        glm::vec3(1.0F),
        glm::vec3(0.25F, 0.5F, 0.75F),
        glm::vec3(0.0031308F, 0.04045F, 0.9F),
    };

    for (const glm::vec3 srgb : values) {
        const glm::vec3 linear = ColorSpace::srgbToLinearRec709(srgb);
        const glm::vec3 roundtrip = ColorSpace::linearRec709ToSrgb(linear);
        EXPECT_NEAR(roundtrip.r, srgb.r, kEpsilon);
        EXPECT_NEAR(roundtrip.g, srgb.g, kEpsilon);
        EXPECT_NEAR(roundtrip.b, srgb.b, kEpsilon);
    }
}

TEST(ColorSpace, PqOetfHasExpectedEndpointsAndMonotonicity) {
    EXPECT_NEAR(ColorSpace::pqOetf(0.0F), 0.0F, kEpsilon);
    EXPECT_NEAR(ColorSpace::pqOetf(1.0F), 1.0F, 1.0e-4F);

    const std::array samples{0.0F, 1.0e-4F, 0.01F, 0.1F, 0.5F, 1.0F};
    float previous = -1.0F;
    for (const float sample : samples) {
        const float encoded = ColorSpace::pqOetf(sample);
        EXPECT_GE(encoded, previous);
        previous = encoded;
    }
}

TEST(ColorSpace, PqOetfVectorOverloadMatchesScalar) {
    const glm::vec3 linear(0.0F, 0.18F, 1.0F);
    const glm::vec3 encoded = ColorSpace::pqOetf(linear);

    EXPECT_NEAR(encoded.r, ColorSpace::pqOetf(linear.r), kEpsilon);
    EXPECT_NEAR(encoded.g, ColorSpace::pqOetf(linear.g), kEpsilon);
    EXPECT_NEAR(encoded.b, ColorSpace::pqOetf(linear.b), kEpsilon);
}

TEST(ColorSpace, Rec2020WhitePointRemainsNeutral) {
    const glm::vec3 white = ColorSpace::rec709ToRec2020(glm::vec3(1.0F));
    EXPECT_NEAR(white.r, 1.0F, kEpsilon);
    EXPECT_NEAR(white.g, 1.0F, kEpsilon);
    EXPECT_NEAR(white.b, 1.0F, kEpsilon);
}

TEST(ColorSpace, Rec2020ConversionPreservesBlack) {
    const glm::vec3 black = ColorSpace::rec709ToRec2020(glm::vec3(0.0F));
    EXPECT_NEAR(glm::length(black), 0.0F, kEpsilon);
}
