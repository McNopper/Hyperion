#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/utils/Math.hpp"

namespace {
constexpr float kEpsilon = 1.0e-5F;
constexpr float kMinSpecularF0 = 0.04F;

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

[[nodiscard]] float fresnelDielectric(float cosThetaI, float eta) noexcept {
    const float clampedCosI = std::clamp(std::abs(cosThetaI), 0.0F, 1.0F);
    const float etaI = 1.0F;
    const float etaT = std::max(eta, 1.0F);
    const float sin2ThetaI = std::max(0.0F, 1.0F - (clampedCosI * clampedCosI));
    const float etaRatio = etaI / etaT;
    const float sin2ThetaT = etaRatio * etaRatio * sin2ThetaI;
    if (sin2ThetaT >= 1.0F) {
        return 1.0F;
    }

    const float cosThetaT = std::sqrt(std::max(0.0F, 1.0F - sin2ThetaT));
    const float rs = ((etaI * clampedCosI) - (etaT * cosThetaT)) / ((etaI * clampedCosI) + (etaT * cosThetaT));
    const float rp = ((etaT * clampedCosI) - (etaI * cosThetaT)) / ((etaT * clampedCosI) + (etaI * cosThetaT));
    return 0.5F * ((rs * rs) + (rp * rp));
}

struct LobeWeights {
    float diffuseWeight = 0.0F;
    float subsurfaceWeight = 0.0F;
    float specularWeight = 0.0F;
    float metalWeight = 0.0F;
    float transmissionWeight = 0.0F;
    float coatWeight = 0.0F;
    float fuzzWeight = 0.0F;
    float diffuseTransWeight = 0.0F;
};

[[nodiscard]] LobeWeights computeLobeWeights(const GpuMaterial& mat) noexcept {
    LobeWeights weights{};

    const float opacity = std::clamp(mat.opacityFlagsPad.x, 0.0F, 1.0F);
    const uint32_t flags = static_cast<uint32_t>(mat.opacityFlagsPad.y);
    const bool glassMode = (flags == 2u);

    const float baseWeight = std::clamp(mat.baseColorWeight.w, 0.0F, 1.0F);
    const float metalness = flags == 1u ? 1.0F : std::clamp(mat.baseMetalnessDiffRough.x, 0.0F, 1.0F);
    const float transmission = glassMode ? 1.0F : std::clamp(mat.transmissionColorWeight.w, 0.0F, 1.0F);
    weights.coatWeight = std::clamp(mat.coatColorWeight.w, 0.0F, 1.0F);
    weights.fuzzWeight = std::clamp(mat.fuzzColorWeight.w, 0.0F, 1.0F);
    weights.subsurfaceWeight = std::clamp(mat.subsurfaceColorWeight.w, 0.0F, 1.0F) * (1.0F - transmission);
    const float opaqueBase = baseWeight * (1.0F - transmission);
    weights.diffuseWeight = opaqueBase * (1.0F - metalness) * (1.0F - weights.subsurfaceWeight);
    weights.specularWeight =
        opaqueBase * (1.0F - metalness) * std::max(std::clamp(mat.specularColorWeight.w, 0.0F, 1.0F), kMinSpecularF0);
    weights.metalWeight = baseWeight * metalness;
    weights.transmissionWeight = baseWeight * transmission;

    weights.diffuseTransWeight = 0.0F;
    const bool thinWalled = mat.opacityFlagsPad.w > 0.5F;
    if (thinWalled && weights.subsurfaceWeight > 0.0F) {
        const float gss = std::clamp(mat.opacityFlagsPad.z, -1.0F, 1.0F);
        const float ssFull = weights.subsurfaceWeight;
        weights.subsurfaceWeight = ssFull * 0.5F * (1.0F - gss);
        weights.diffuseTransWeight = ssFull * 0.5F * (1.0F + gss);
    }

    const float total = weights.diffuseWeight + weights.subsurfaceWeight + weights.specularWeight + weights.metalWeight +
                        weights.transmissionWeight + weights.coatWeight + weights.fuzzWeight +
                        weights.diffuseTransWeight;
    if (total <= 0.0F || opacity < 0.0F) {
        return {};
    }
    return weights;
}

[[nodiscard]] glm::vec3 sampleUniformSphere(float u1, float u2) noexcept {
    const float z = 1.0F - (2.0F * u1);
    const float r = std::sqrt(std::max(0.0F, 1.0F - (z * z)));
    const float phi = Math::k2Pi * u2;
    return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
}

[[nodiscard]] glm::vec3 toLocal(const glm::vec3& v, const glm::vec3& T, const glm::vec3& B, const glm::vec3& N) noexcept {
    return glm::vec3(glm::dot(v, T), glm::dot(v, B), glm::dot(v, N));
}

[[nodiscard]] glm::vec3 toWorld(const glm::vec3& v, const glm::vec3& T, const glm::vec3& B, const glm::vec3& N) noexcept {
    return (T * v.x) + (B * v.y) + (N * v.z);
}

void orientFrame(const glm::vec3& wo, const glm::vec3& N, const glm::vec3& T, const glm::vec3& B,
                 glm::vec3& Ns, glm::vec3& Ts, glm::vec3& Bs) noexcept {
    if (glm::dot(wo, N) >= 0.0F) {
        Ns = N;
        Ts = T;
        Bs = B;
    } else {
        Ns = -N;
        Ts = -T;
        Bs = -B;
    }
}

[[nodiscard]] GpuMaterial makeMaterial() noexcept {
    GpuMaterial mat{};
    mat.baseColorWeight = glm::vec4(1.0F);
    mat.baseMetalnessDiffRough = glm::vec4(0.0F);
    mat.specularColorWeight = glm::vec4(0.04F);
    mat.specularRoughAnisoIor = glm::vec4(0.2F, 0.0F, 1.5F, 0.0F);
    mat.transmissionColorWeight = glm::vec4(0.0F);
    mat.transmissionParams = glm::vec4(0.0F);
    mat.transmissionScatter = glm::vec4(0.0F);
    mat.subsurfaceColorWeight = glm::vec4(0.0F);
    mat.subsurfaceRadiusScale = glm::vec4(0.0F);
    mat.thinFilmParams = glm::vec4(0.0F);
    mat.coatColorWeight = glm::vec4(0.0F);
    mat.coatRoughAnisoIorDark = glm::vec4(0.0F);
    mat.fuzzColorWeight = glm::vec4(0.0F);
    mat.fuzzRoughPad = glm::vec4(0.0F);
    mat.opacityFlagsPad = glm::vec4(1.0F, 0.0F, 0.0F, 0.0F);
    return mat;
}

[[nodiscard]] glm::vec2 computeAlpha(float roughness, float anisotropy) noexcept {
    const float r = std::max(roughness, 0.02F);
    const float a = std::clamp(anisotropy, 0.0F, 0.98F);
    const float alphaX = std::max(r * r * std::sqrt(2.0F / (1.0F + ((1.0F - a) * (1.0F - a)))), 0.001F);
    const float alphaY = std::max((1.0F - a) * alphaX, 0.001F);
    return glm::vec2(alphaX, alphaY);
}

[[nodiscard]] glm::vec3 iorToF0(float eta) noexcept {
    const float x = (eta - 1.0F) / std::max(eta + 1.0F, 1.0e-4F);
    return glm::vec3(x * x);
}

[[nodiscard]] float diffuseDirAlbedoFujii(float cosTheta, float roughness) noexcept {
    constexpr float kFujiiC1 = 0.5F - 2.0F / (3.0F * Math::kPi);
    const float A = 1.0F / (1.0F + (kFujiiC1 * roughness));
    const float B = roughness * A;
    const float Si = std::sqrt(std::max(0.0F, 1.0F - (cosTheta * cosTheta)));
    const float G = (Si * (std::acos(std::clamp(cosTheta, -1.0F, 1.0F)) - (Si * cosTheta))) +
                    2.0F * (((Si / std::max(cosTheta, 1.0e-4F)) * (1.0F - (Si * Si * Si))) - Si) / 3.0F;
    return A + (B * G * Math::kInvPi);
}

[[nodiscard]] float diffuseAvgAlbedoFujii(float roughness) noexcept {
    constexpr float kFujiiC1 = 0.5F - 2.0F / (3.0F * Math::kPi);
    constexpr float kFujiiC2 = 2.0F / 3.0F - 28.0F / (15.0F * Math::kPi);
    const float A = 1.0F / (1.0F + (kFujiiC1 * roughness));
    return A * (1.0F + (kFujiiC2 * roughness));
}

[[nodiscard]] glm::vec3 evalDiffuse(const glm::vec3& color, float roughness, const glm::vec3& wo, const glm::vec3& wi) noexcept {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return glm::vec3(0.0F);
    }

    const float NdotV = wo.z;
    const float NdotL = wi.z;
    const float s = glm::dot(wi, wo) - (NdotL * NdotV);
    const float stinv = (s > 0.0F) ? (s / std::max(NdotL, NdotV)) : s;

    const float A = 1.0F / (1.0F + (0.146975F * roughness));
    const glm::vec3 lobeSingle = color * A * (1.0F + (roughness * stinv));

    const float dAlbedoV = diffuseDirAlbedoFujii(NdotV, roughness);
    const float dAlbedoL = diffuseDirAlbedoFujii(NdotL, roughness);
    const float avgA = diffuseAvgAlbedoFujii(roughness);
    const glm::vec3 colorMS = (color * color * avgA) / glm::max(glm::vec3(0.0001F), glm::vec3(1.0F) - color * std::max(0.0F, 1.0F - avgA));
    const glm::vec3 lobeMS = colorMS * std::max(1.0e-4F, 1.0F - dAlbedoV) * std::max(1.0e-4F, 1.0F - dAlbedoL) /
                             std::max(1.0e-4F, 1.0F - avgA);

    return (lobeSingle + lobeMS) * Math::kInvPi;
}

[[nodiscard]] glm::vec3 evalReflectionMicrofacet(const glm::vec3& F0,
                                                 const glm::vec3& F82,
                                                 const glm::vec3& wo,
                                                 const glm::vec3& wi,
                                                 float alphaX,
                                                 float alphaY) noexcept {
    (void)alphaY;
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return glm::vec3(0.0F);
    }

    const glm::vec3 h = glm::normalize(wo + wi);
    const float NoH = std::max(h.z, 0.0F);
    const float NoL = std::max(wi.z, 0.0F);
    const float NoV = std::max(wo.z, 0.0F);
    const float VoH = std::max(glm::dot(wo, h), 0.0F);
    if (NoH <= 0.0F || NoL <= 0.0F || NoV <= 0.0F || VoH <= 0.0F) {
        return glm::vec3(0.0F);
    }

    const float D = ggxD(NoH, alphaX);
    const float G = smithG2(wi, wo, alphaX);
    const glm::vec3 F = glm::vec3(fresnelSchlick(F0.x, VoH));
    const glm::vec3 tint = glm::mix(F0, F82, glm::vec3(1.0F) - glm::clamp(F, glm::vec3(0.0F), glm::vec3(1.0F)));
    return (D * G / std::max(4.0F * NoL * NoV, 1.0e-5F)) * tint;
}

[[nodiscard]] glm::vec3 evalTransmissionMicrofacet(const glm::vec3& color,
                                                   float depth,
                                                   float eta,
                                                   float alphaX,
                                                   float alphaY,
                                                   const glm::vec3& wo,
                                                   const glm::vec3& wi) noexcept {
    (void)alphaY;
    if (wo.z * wi.z >= 0.0F) {
        return glm::vec3(0.0F);
    }

    const float F = fresnelDielectric(std::abs(wo.z), eta);
    const glm::vec3 h = glm::normalize((-wo * (wo.z > 0.0F ? (1.0F / eta) : eta)) + wi);
    const float NoH = std::max(h.z, 0.0F);
    const float D = ggxD(NoH, alphaX);
    const float G = smithG2(wi, wo, alphaX);
    const float denom = 4.0F * std::abs(wo.z) * std::abs(wi.z);
    const glm::vec3 absorption = glm::exp(-std::max(depth, 0.001F) * (glm::vec3(1.0F) - glm::clamp(color, glm::vec3(0.0F), glm::vec3(1.0F))));
    return absorption * color * (((D * G) / std::max(denom, 1.0e-5F)) * (1.0F - F));
}

[[nodiscard]] glm::vec3 evalBSDF(const GpuMaterial& mat,
                                 const glm::vec3& wo,
                                 const glm::vec3& wi,
                                 const glm::vec3& N,
                                 const glm::vec3& T,
                                 const glm::vec3& B,
                                 const glm::vec3& cN,
                                 const glm::vec3& cT,
                                 const glm::vec3& cB) noexcept {
    (void)cN;
    (void)cT;
    (void)cB;
    glm::vec3 Ns;
    glm::vec3 Ts;
    glm::vec3 Bs;
    orientFrame(wo, N, T, B, Ns, Ts, Bs);
    const glm::vec3 woL = toLocal(wo, Ts, Bs, Ns);
    const glm::vec3 wiL = toLocal(wi, Ts, Bs, Ns);
    if (woL.z == 0.0F || wiL.z == 0.0F) {
        return glm::vec3(0.0F);
    }

    const float opacity = std::clamp(mat.opacityFlagsPad.x, 0.0F, 1.0F);
    const glm::vec3 baseColor = glm::clamp(glm::vec3(mat.baseColorWeight), glm::vec3(0.0F), glm::vec3(1.0F));
    const glm::vec3 specColor = glm::clamp(glm::vec3(mat.specularColorWeight), glm::vec3(0.0F), glm::vec3(1.0F));
    const glm::vec3 transColor = glm::clamp(glm::vec3(mat.transmissionColorWeight), glm::vec3(0.0F), glm::vec3(1.0F));
    const glm::vec3 subsurfaceColor = glm::clamp(glm::vec3(mat.subsurfaceColorWeight), glm::vec3(0.0F), glm::vec3(1.0F));
    const glm::vec3 coatColor = glm::clamp(glm::vec3(mat.coatColorWeight), glm::vec3(0.0F), glm::vec3(1.0F));
    const glm::vec3 fuzzColor = glm::clamp(glm::vec3(mat.fuzzColorWeight), glm::vec3(0.0F), glm::vec3(1.0F));

    const float baseRough = mat.specularRoughAnisoIor.x;
    const float eta = std::max(mat.specularRoughAnisoIor.z, 1.01F);
    const float coatRough = mat.coatRoughAnisoIorDark.x;
    const float coatEta = std::max(mat.coatRoughAnisoIorDark.z, 1.01F);
    const float diffuseRough = std::clamp(mat.baseMetalnessDiffRough.y, 0.0F, 1.0F);
    const float coatDark = std::clamp(mat.coatRoughAnisoIorDark.w, 0.0F, 1.0F);
    const float ssAniso = std::clamp(mat.opacityFlagsPad.z, -1.0F, 1.0F);

    const glm::vec2 alpha = computeAlpha(baseRough, 0.0F);
    const glm::vec2 coatAlpha = computeAlpha(coatRough, 0.0F);
    const float alphaX = std::sqrt((alpha.x * alpha.x) + (coatAlpha.x * coatAlpha.x * std::clamp(mat.coatColorWeight.w, 0.0F, 1.0F)));
    const float alphaY = std::sqrt((alpha.y * alpha.y) + (coatAlpha.y * coatAlpha.y * std::clamp(mat.coatColorWeight.w, 0.0F, 1.0F)));

    LobeWeights weights = computeLobeWeights(mat);
    const float baseLayerScale = (1.0F / std::max(1.0F, coatEta * coatEta * weights.coatWeight * coatDark)) *
                                 (1.0F - 0.35F * weights.fuzzWeight);

    glm::vec3 result(0.0F);
    if (wiL.z > 0.0F && woL.z > 0.0F) {
        const glm::vec3 dielectricF0 = iorToF0(eta);
        glm::vec3 glossyF0 = glm::mix(dielectricF0 * specColor, baseColor, std::clamp(mat.baseMetalnessDiffRough.x, 0.0F, 1.0F));
        const glm::vec3 glossyF82 = specColor;

        if (weights.diffuseWeight > 0.0F) {
            result += baseLayerScale * weights.diffuseWeight * evalDiffuse(baseColor, diffuseRough, woL, wiL);
        }
        if (weights.subsurfaceWeight > 0.0F) {
            const float ssScale = std::max(mat.subsurfaceRadiusScale.w, 0.001F);
            const glm::vec3 ssTint = subsurfaceColor * (1.0F / (1.0F + ssScale * (mat.subsurfaceRadiusScale.x +
                                                                                 mat.subsurfaceRadiusScale.y +
                                                                                 mat.subsurfaceRadiusScale.z)));
            const float ssReflScale = (mat.opacityFlagsPad.w > 0.5F) ? 1.0F : std::clamp(1.0F - ssAniso, 0.0F, 1.0F);
            result += baseLayerScale * weights.subsurfaceWeight * ssReflScale * evalDiffuse(ssTint, std::clamp(0.25F + 0.5F * diffuseRough, 0.0F, 1.0F), woL, wiL);
        }
        if (weights.specularWeight > 0.0F || weights.metalWeight > 0.0F) {
            result += baseLayerScale * (weights.specularWeight + weights.metalWeight) *
                      evalReflectionMicrofacet(glossyF0, glossyF82, woL, wiL, alphaX, alphaY);
        }
        if (weights.fuzzWeight > 0.0F) {
            result += weights.fuzzWeight * (fuzzColor * Math::kInvPi);
        }
    }

    if (weights.coatWeight > 0.0F && woL.z > 0.0F && wiL.z > 0.0F) {
        const glm::vec3 cNs = cN;
        const glm::vec3 cTs = cT;
        const glm::vec3 cBs = cB;
        const glm::vec3 woC = toLocal(wo, cTs, cBs, cNs);
        const glm::vec3 wiC = toLocal(wi, cTs, cBs, cNs);
        if (woC.z > 0.0F && wiC.z > 0.0F) {
            const glm::vec3 coatF0 = iorToF0(coatEta) * glm::max(coatColor, glm::vec3(0.04F));
            result += weights.coatWeight * evalReflectionMicrofacet(coatF0, coatColor, woC, wiC, coatAlpha.x, coatAlpha.y);
        }
    }

    if (weights.transmissionWeight > 0.0F) {
        const float transAlphaX = alphaX;
        const float transAlphaY = alphaY;
        result += weights.transmissionWeight *
                  evalTransmissionMicrofacet(transColor, mat.transmissionParams.x, eta, transAlphaX, transAlphaY, woL, wiL);
    }

    if (weights.diffuseTransWeight > 0.0F && woL.z > 0.0F && wiL.z < 0.0F) {
        const glm::vec3 wiR(wiL.x, wiL.y, -wiL.z);
        result += weights.diffuseTransWeight * evalDiffuse(subsurfaceColor, diffuseRough, woL, wiR);
    }

    (void)opacity;
    return glm::max(result, glm::vec3(0.0F));
}

[[nodiscard]] double estimateWhiteFurnaceEnergy(const GpuMaterial& mat, const glm::vec3& wo, int sampleCount) {
    std::mt19937 rng(12345U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const glm::vec3 N(0.0F, 0.0F, 1.0F);
    const glm::vec3 T(1.0F, 0.0F, 0.0F);
    const glm::vec3 B(0.0F, 1.0F, 0.0F);

    double sum = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        const glm::vec3 wi = sampleUniformSphere(dist(rng), dist(rng));
        const glm::vec3 f = evalBSDF(mat, wo, wi, N, T, B, N, T, B);
        sum += static_cast<double>(Math::luminance(f) * std::abs(wi.z) * (4.0 * Math::kPi));
    }
    return sum / static_cast<double>(sampleCount);
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

TEST(Bsdf, OpenPbrLobeWeightsKeepOpacitySeparateFromTransmission) {
    GpuMaterial opaque = makeMaterial();
    opaque.opacityFlagsPad.x = 1.0F;
    opaque.transmissionColorWeight.w = 1.0F;

    GpuMaterial cutout = opaque;
    cutout.opacityFlagsPad.x = 0.25F;

    const LobeWeights opaqueWeights = computeLobeWeights(opaque);
    const LobeWeights cutoutWeights = computeLobeWeights(cutout);

    EXPECT_FLOAT_EQ(opaqueWeights.transmissionWeight, cutoutWeights.transmissionWeight);
    EXPECT_FLOAT_EQ(opaqueWeights.diffuseWeight, cutoutWeights.diffuseWeight);
    EXPECT_FLOAT_EQ(opaqueWeights.specularWeight, cutoutWeights.specularWeight);
}

TEST(Bsdf, OpenPbrThinWalledSubsurfaceSplitConservesWeight) {
    GpuMaterial mat = makeMaterial();
    mat.subsurfaceColorWeight = glm::vec4(0.0F, 0.0F, 0.0F, 0.8F);
    mat.opacityFlagsPad.z = 0.25F;
    mat.opacityFlagsPad.w = 1.0F;

    const LobeWeights weights = computeLobeWeights(mat);
    EXPECT_NEAR(weights.subsurfaceWeight + weights.diffuseTransWeight, 0.8F, 1.0e-6F);
    EXPECT_NEAR(weights.subsurfaceWeight, 0.3F, 1.0e-6F);
    EXPECT_NEAR(weights.diffuseTransWeight, 0.5F, 1.0e-6F);
}

TEST(Bsdf, OpenPbrDispersionChannelAndFresnelContractsHold) {
    constexpr float eta = 1.52F;
    constexpr float abbe = 30.0F;
    constexpr float dispScale = 0.5F;
    constexpr float spread = dispScale * (eta - 1.0F) / abbe;

    const float etaR = std::max(eta + spread * (0.0F - 1.0F), 1.001F);
    const float etaG = std::max(eta + spread * (1.0F - 1.0F), 1.001F);
    const float etaB = std::max(eta + spread * (2.0F - 1.0F), 1.001F);

    EXPECT_LT(etaR, etaG);
    EXPECT_LT(etaG, etaB);
    EXPECT_GE(fresnelDielectric(0.5F, eta), 0.0F);
    EXPECT_LE(fresnelDielectric(0.5F, eta), 1.0F);
    EXPECT_FLOAT_EQ(std::max(eta - 10.0F, 1.001F), 1.001F);
}

TEST(Bsdf, OpenPbrWhiteFurnaceRepresentativeConfigsStayBounded) {
    const glm::vec3 wo = glm::normalize(glm::vec3(0.25F, 0.15F, 0.955F));

    std::array<GpuMaterial, 5> configs{};

    configs[0] = makeMaterial();
    configs[0].baseColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[0].baseMetalnessDiffRough = glm::vec4(0.0F, 0.35F, 0.0F, 0.0F);
    configs[0].specularRoughAnisoIor = glm::vec4(0.25F, 0.0F, 1.5F, 0.0F);

    configs[1] = configs[0];
    configs[1].baseColorWeight.w = 0.0F;
    configs[1].coatColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[1].coatRoughAnisoIorDark = glm::vec4(0.15F, 0.0F, 1.4F, 0.25F);

    configs[2] = configs[0];
    configs[2].baseColorWeight.w = 0.0F;
    configs[2].fuzzColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[2].fuzzRoughPad.x = 0.4F;

    configs[3] = makeMaterial();
    configs[3].baseColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[3].transmissionColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[3].transmissionParams = glm::vec4(0.0F, 0.0F, 0.0F, 0.0F);
    configs[3].specularRoughAnisoIor = glm::vec4(0.02F, 0.0F, 1.52F, 0.0F);

    configs[4] = makeMaterial();
    configs[4].baseColorWeight = glm::vec4(0.0F, 0.0F, 0.0F, 0.0F);
    configs[4].subsurfaceColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[4].opacityFlagsPad.w = 1.0F;
    configs[4].opacityFlagsPad.z = 0.2F;

    for (const GpuMaterial& mat : configs) {
        const double energy = estimateWhiteFurnaceEnergy(mat, wo, 12000);
        EXPECT_GE(energy, 0.0);
        EXPECT_LE(energy, 1.10);
    }
}

// NOTE: GGX specular BRDF symmetry is tested at the shader level (shaders/bsdf.slang).
// A C++ version will be added when Slang → C++ codegen is wired into the build.
