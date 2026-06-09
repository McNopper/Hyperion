#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "harmonia/utils/Math.hpp"

namespace {
constexpr float kEpsilon = 1.0e-5F;

[[nodiscard]] glm::vec3 sampleUniformHemisphere(float u1, float u2) noexcept {
    const float z = u1;
    const float r = std::sqrt(std::max(0.0F, 1.0F - (z * z)));
    const float phi = Math::k2Pi * u2;
    return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
}

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

[[nodiscard]] float smithG2(glm::vec3 l, glm::vec3 v, float alpha) noexcept {
    return smithG1(std::max(l.z, 0.0F), alpha) * smithG1(std::max(v.z, 0.0F), alpha);
}

[[nodiscard]] float fresnelSchlick(float f0, float cosTheta) noexcept {
    const float oneMinusCos = 1.0F - std::clamp(cosTheta, 0.0F, 1.0F);
    const float factor = oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos * oneMinusCos;
    return f0 + ((1.0F - f0) * factor);
}

[[nodiscard]] float orenNayarDiffuse(glm::vec3 wo, glm::vec3 wi, float roughness, float albedo) noexcept {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return 0.0F;
    }

    const float sigma2 = roughness * roughness;
    const float a = 1.0F - (0.5F * sigma2 / (sigma2 + 0.33F));
    const float b = 0.45F * sigma2 / (sigma2 + 0.09F);

    const float sinThetaI = std::sqrt(std::max(0.0F, 1.0F - (wi.z * wi.z)));
    const float sinThetaO = std::sqrt(std::max(0.0F, 1.0F - (wo.z * wo.z)));

    float maxCos = 0.0F;
    if (sinThetaI > 0.0F && sinThetaO > 0.0F) {
        const glm::vec2 wiTangent = glm::normalize(glm::vec2(wi.x, wi.y));
        const glm::vec2 woTangent = glm::normalize(glm::vec2(wo.x, wo.y));
        maxCos = std::max(glm::dot(wiTangent, woTangent), 0.0F);
    }

    const bool incidentIsSteeper = wi.z < wo.z;
    const float sinAlpha = incidentIsSteeper ? sinThetaO : sinThetaI;
    const float tanBeta =
        incidentIsSteeper ? (sinThetaI / std::max(wi.z, kEpsilon)) : (sinThetaO / std::max(wo.z, kEpsilon));
    return (albedo * Math::kInvPi) * (a + (b * maxCos * sinAlpha * tanBeta));
}

// NOTE: ggxSpecularBrdf (and tests for GGX BRDF symmetry / reciprocity) live in
// shaders/bsdf.slang only. Tests will be added when Slang → C++ codegen is wired in.
} // namespace

TEST(Bsdf, GgxDNormalization) {
    constexpr float roughness = 0.5F;
    constexpr int sampleCount = 50000;
    std::mt19937 rng(123U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);

    double integral = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 m = sampleUniformHemisphere(dist(rng), dist(rng));
        integral += ggxD(m.z, roughness) * m.z / Math::kInv2Pi;
    }
    integral /= static_cast<double>(sampleCount);

    EXPECT_NEAR(static_cast<float>(integral), 1.0F, 2.0e-2F);
}

TEST(Bsdf, GgxSmithG2IsSymmetric) {
    constexpr float roughness = 0.35F;
    std::mt19937 rng(456U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);

    for (int i = 0; i < 2000; ++i) {
        const glm::vec3 l = sampleUniformHemisphere(dist(rng), dist(rng));
        const glm::vec3 v = sampleUniformHemisphere(dist(rng), dist(rng));
        EXPECT_NEAR(smithG2(l, v, roughness), smithG2(v, l, roughness), 1.0e-6F);
    }
}

TEST(Bsdf, FresnelEnergyRemainsInUnitRange) {
    constexpr float f0 = 0.04F;
    for (int i = 0; i <= 1000; ++i) {
        const float cosTheta = static_cast<float>(i) / 1000.0F;
        const float value = fresnelSchlick(f0, cosTheta);
        EXPECT_GE(value, 0.0F);
        EXPECT_LE(value, 1.0F);
    }
}

TEST(Bsdf, DiffuseEonReflectanceDoesNotExceedOne) {
    constexpr int sampleCount = 20000;
    std::mt19937 rng(789U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const std::array outgoingDirections{
        glm::vec3(0.0F, 0.0F, 1.0F),
        glm::normalize(glm::vec3(0.5F, 0.0F, 0.8660254F)),
        glm::normalize(glm::vec3(0.8F, 0.0F, 0.6F)),
    };

    for (const float roughness : {0.0F, 0.25F, 0.5F, 0.9F}) {
        for (const glm::vec3 wo : outgoingDirections) {
            double reflectance = 0.0;
            for (int i = 0; i < sampleCount; ++i) {
                const glm::vec3 wi = sampleUniformHemisphere(dist(rng), dist(rng));
                const float brdf = orenNayarDiffuse(wo, wi, roughness, 1.0F);
                reflectance += brdf * wi.z / Math::kInv2Pi;
            }
            reflectance /= static_cast<double>(sampleCount);
            EXPECT_LE(static_cast<float>(reflectance), 1.0F + 2.0e-2F);
        }
    }
}

// NOTE: GGX specular BRDF symmetry is tested at the shader level (shaders/bsdf.slang).
// A C++ version will be added when Slang → C++ codegen is wired into the build.
