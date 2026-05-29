#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <random>
#include <unordered_set>
#include <vector>

#include "hyperion/utils/Math.hpp"

namespace {
constexpr float kEpsilon = 1.0e-5F;
constexpr float kMonteCarloTolerance = 2.0e-2F;

[[nodiscard]] glm::vec3 sampleUniformHemisphere(float u1, float u2) noexcept {
    const float z = u1;
    const float r = std::sqrt(std::max(0.0F, 1.0F - (z * z)));
    const float phi = Math::k2Pi * u2;
    return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
}

[[nodiscard]] glm::mat3 buildTbn(glm::vec3 n) noexcept {
    n = glm::normalize(n);
    const glm::vec3 up = std::abs(n.z) < 0.999F ? glm::vec3(0.0F, 0.0F, 1.0F) : glm::vec3(0.0F, 1.0F, 0.0F);
    const glm::vec3 t = glm::normalize(glm::cross(up, n));
    const glm::vec3 b = glm::cross(n, t);
    return glm::mat3(t, b, n);
}

[[nodiscard]] glm::vec3 fresnelSchlick(glm::vec3 f0, float cosTheta) noexcept {
    const float oneMinusCos = 1.0F - std::clamp(cosTheta, 0.0F, 1.0F);
    const float factor = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return f0 + (glm::vec3(1.0F) - f0) * factor;
}

// NOTE: fresnelDielectric and fresnelDielectricParallel live in shaders/math.slang only.
// Tests for those functions will be added when Slang → C++ codegen is wired into the build.

[[nodiscard]] float ggxD(float nDotH, float alpha) noexcept {
    if (nDotH <= 0.0F) {
        return 0.0F;
    }
    const float alpha2 = alpha * alpha;
    const float denom = (nDotH * nDotH) * (alpha2 - 1.0F) + 1.0F;
    return alpha2 / (Math::kPi * denom * denom);
}

[[nodiscard]] float smithLambdaGgx(float nDotV, float alpha) noexcept {
    if (nDotV <= 0.0F) {
        return std::numeric_limits<float>::infinity();
    }
    const float cos2 = nDotV * nDotV;
    const float tan2 = (1.0F - cos2) / cos2;
    return 0.5F * (-1.0F + std::sqrt(1.0F + (alpha * alpha * tan2)));
}

[[nodiscard]] float smithG1(float nDotV, float alpha) noexcept {
    if (nDotV <= 0.0F) {
        return 0.0F;
    }
    return 1.0F / (1.0F + smithLambdaGgx(nDotV, alpha));
}

[[nodiscard]] float ggxG2(glm::vec3 l, glm::vec3 v, float alpha) noexcept {
    const float nDotL = std::max(l.z, 0.0F);
    const float nDotV = std::max(v.z, 0.0F);
    return smithG1(nDotL, alpha) * smithG1(nDotV, alpha);
}

[[nodiscard]] glm::vec3 sampleGgxVndf(glm::vec3 v, float alpha, glm::vec2 u) noexcept {
    const glm::vec3 vh = glm::normalize(glm::vec3(alpha * v.x, alpha * v.y, v.z));
    const float lensq = (vh.x * vh.x) + (vh.y * vh.y);
    const glm::vec3 t1 = lensq > 0.0F ? glm::vec3(-vh.y, vh.x, 0.0F) / std::sqrt(lensq) : glm::vec3(1.0F, 0.0F, 0.0F);
    const glm::vec3 t2 = glm::cross(vh, t1);

    const float r = std::sqrt(u.x);
    const float phi = Math::k2Pi * u.y;
    float p1 = r * std::cos(phi);
    float p2 = r * std::sin(phi);
    const float s = 0.5F * (1.0F + vh.z);
    p2 = ((1.0F - s) * std::sqrt(std::max(0.0F, 1.0F - (p1 * p1)))) + (s * p2);

    const float p3 = std::sqrt(std::max(0.0F, 1.0F - (p1 * p1) - (p2 * p2)));
    const glm::vec3 nh = (p1 * t1) + (p2 * t2) + (p3 * vh);
    return glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, std::max(0.0F, nh.z)));
}

[[nodiscard]] float ggxVndfPdf(glm::vec3 v, glm::vec3 m, float alpha) noexcept {
    const float nDotV = std::max(v.z, 0.0F);
    const float nDotM = std::max(m.z, 0.0F);
    const float vDotM = std::max(glm::dot(v, m), 0.0F);
    if (nDotV <= 0.0F || nDotM <= 0.0F || vDotM <= 0.0F) {
        return 0.0F;
    }
    return ggxD(nDotM, alpha) * smithG1(nDotV, alpha) * vDotM / nDotV;
}

[[nodiscard]] uint32_t wangHash(uint32_t seed) noexcept {
    seed = (seed ^ 61U) ^ (seed >> 16U);
    seed *= 9U;
    seed ^= seed >> 4U;
    seed *= 0x27d4eb2dU;
    seed ^= seed >> 15U;
    return seed;
}

[[nodiscard]] float randFloat(uint32_t& state) noexcept {
    state = wangHash(state);
    return static_cast<float>(state) * (1.0F / 4294967296.0F);
}
} // namespace

TEST(Math, BuildTbnIsOrthonormalAndStable) {
    const std::array normals{
        glm::normalize(glm::vec3(0.0F, 1.0F, 0.0F)),
        glm::normalize(glm::vec3(1.0F, 0.0F, 0.0F)),
        glm::normalize(glm::vec3(0.0F, 0.0F, 1.0F)),
        glm::normalize(glm::vec3(0.577F, 0.577F, 0.577F)),
    };

    for (const glm::vec3 n : normals) {
        const glm::mat3 tbn = buildTbn(n);
        const glm::vec3 t = tbn[0];
        const glm::vec3 b = tbn[1];
        const glm::vec3 z = tbn[2];

        EXPECT_NEAR(glm::length(t), 1.0F, kEpsilon);
        EXPECT_NEAR(glm::length(b), 1.0F, kEpsilon);
        EXPECT_NEAR(glm::length(z), 1.0F, kEpsilon);
        EXPECT_NEAR(glm::dot(t, b), 0.0F, kEpsilon);
        EXPECT_NEAR(glm::dot(t, z), 0.0F, kEpsilon);
        EXPECT_NEAR(glm::dot(b, z), 0.0F, kEpsilon);
        EXPECT_NEAR(glm::determinant(tbn), 1.0F, 5.0e-4F);
        EXPECT_NEAR(glm::length(z - n), 0.0F, kEpsilon);
    }
}

TEST(Math, FresnelSchlickHasExpectedLimitsAndMonotonicity) {
    const glm::vec3 f0(0.04F, 0.25F, 0.9F);

    const glm::vec3 normalIncidence = fresnelSchlick(f0, 1.0F);
    const glm::vec3 grazing = fresnelSchlick(f0, 0.0F);

    EXPECT_NEAR(glm::length(normalIncidence - f0), 0.0F, kEpsilon);
    EXPECT_NEAR(grazing.r, 1.0F, kEpsilon);
    EXPECT_NEAR(grazing.g, 1.0F, kEpsilon);
    EXPECT_NEAR(grazing.b, 1.0F, kEpsilon);

    glm::vec3 previous = fresnelSchlick(f0, 1.0F);
    for (int i = 1; i <= 16; ++i) {
        const float cosTheta = 1.0F - (static_cast<float>(i) / 16.0F);
        const glm::vec3 current = fresnelSchlick(f0, cosTheta);
        EXPECT_GE(current.r + kEpsilon, previous.r);
        EXPECT_GE(current.g + kEpsilon, previous.g);
        EXPECT_GE(current.b + kEpsilon, previous.b);
        EXPECT_LE(current.r, 1.0F + kEpsilon);
        EXPECT_LE(current.g, 1.0F + kEpsilon);
        EXPECT_LE(current.b, 1.0F + kEpsilon);
        previous = current;
    }
}

TEST(Math, GgxDIntegratesToOneOverHemisphere) {
    std::mt19937 rng(12345U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);

    constexpr float alpha = 0.5F;
    constexpr int sampleCount = 50000;
    const float pdf = Math::kInv2Pi;

    double estimate = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 h = sampleUniformHemisphere(dist(rng), dist(rng));
        estimate += ggxD(h.z, alpha) * h.z / pdf;
    }
    estimate /= static_cast<double>(sampleCount);

    EXPECT_NEAR(static_cast<float>(estimate), 1.0F, kMonteCarloTolerance);
}

TEST(Math, GgxG2StaysInRangeAndIsSymmetric) {
    constexpr float alpha = 0.5F;
    std::mt19937 rng(4242U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);

    for (int i = 0; i < 1000; ++i) {
        const glm::vec3 l = sampleUniformHemisphere(dist(rng), dist(rng));
        const glm::vec3 v = sampleUniformHemisphere(dist(rng), dist(rng));
        const float gLv = ggxG2(l, v, alpha);
        const float gVl = ggxG2(v, l, alpha);
        EXPECT_GE(gLv, 0.0F);
        EXPECT_LE(gLv, 1.0F + kEpsilon);
        EXPECT_NEAR(gLv, gVl, 1.0e-6F);
    }

    EXPECT_NEAR(ggxG2(glm::vec3(0.0F, 0.0F, 1.0F), glm::vec3(0.0F, 0.0F, 1.0F), alpha), 1.0F, kEpsilon);
}

TEST(Math, SampleGgxVndfProducesNormalizedPdf) {
    constexpr float alpha = 0.5F;
    const glm::vec3 v = glm::normalize(glm::vec3(0.3F, -0.2F, 0.9327379F));

    std::mt19937 rng(7U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);

    constexpr int integralSampleCount = 10000;
    const float hemispherePdf = Math::kInv2Pi;
    double pdfIntegral = 0.0;
    double pdfWeightedMeanCos = 0.0;
    for (int i = 0; i < integralSampleCount; ++i) {
        const glm::vec3 m = sampleUniformHemisphere(dist(rng), dist(rng));
        const float pdf = ggxVndfPdf(v, m, alpha);
        pdfIntegral += pdf / hemispherePdf;
        pdfWeightedMeanCos += (m.z * pdf) / hemispherePdf;
    }
    pdfIntegral /= static_cast<double>(integralSampleCount);
    pdfWeightedMeanCos /= static_cast<double>(integralSampleCount);

    EXPECT_NEAR(static_cast<float>(pdfIntegral), 1.0F, 2.5e-2F);

    constexpr int sampleCount = 10000;
    double sampledMeanCos = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 m = sampleGgxVndf(v, alpha, glm::vec2(dist(rng), dist(rng)));
        EXPECT_NEAR(glm::length(m), 1.0F, 1.0e-4F);
        EXPECT_GE(m.z, -kEpsilon);
        sampledMeanCos += m.z;
    }
    sampledMeanCos /= static_cast<double>(sampleCount);

    EXPECT_NEAR(static_cast<float>(sampledMeanCos), static_cast<float>(pdfWeightedMeanCos), 3.0e-2F);
}

TEST(Math, WangHashSeparatesConsecutiveSeeds) {
    std::unordered_set<uint32_t> values;
    values.reserve(1000);

    for (uint32_t seed = 0; seed < 1000U; ++seed) {
        const uint32_t hash = wangHash(seed);
        EXPECT_TRUE(values.insert(hash).second) << "Duplicate hash for seed " << seed;
    }
}

TEST(Math, RandFloatStaysWithinUnitInterval) {
    uint32_t state = 1U;
    for (int i = 0; i < 10000; ++i) {
        const float value = randFloat(state);
        EXPECT_GE(value, 0.0F);
        EXPECT_LT(value, 1.0F);
    }
}
