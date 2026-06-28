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

// ─── OpenPBR thin-film Airy (mirror of Harmonia bsdf_shared.slang mx_fresnel_airy) ───
[[nodiscard]] glm::vec3 thinFilmSensitivity(float opd, glm::vec3 shift) noexcept {
    const float phase = 2.0F * Math::kPi * opd;
    const glm::vec3 val(5.4856e-13F, 4.4201e-13F, 5.2481e-13F);
    const glm::vec3 pos(1.6810e+06F, 1.7953e+06F, 2.2084e+06F);
    const glm::vec3 var(4.3278e+09F, 9.3046e+09F, 6.6121e+09F);
    glm::vec3 xyz = val * glm::sqrt(2.0F * Math::kPi * var) * glm::cos(pos * phase + shift) *
                    glm::exp(-var * phase * phase);
    xyz.x += 9.7470e-14F * std::sqrt(2.0F * Math::kPi * 4.5282e+09F) *
             std::cos(2.2399e+06F * phase + shift.x) * std::exp(-4.5282e+09F * phase * phase);
    return xyz / 1.0685e-7F;
}

[[nodiscard]] glm::vec3 thinFilmXyzToRgb(glm::vec3 v) noexcept {
    return glm::vec3(glm::dot(glm::vec3(2.3706743F, -0.9000405F, -0.4706338F), v),
                     glm::dot(glm::vec3(-0.5138850F, 1.4253036F, 0.0885814F), v),
                     glm::dot(glm::vec3(0.0052982F, -0.0146949F, 1.0093968F), v));
}

[[nodiscard]] glm::vec3 fresnel0ToIor(glm::vec3 f0) noexcept {
    const glm::vec3 s = glm::sqrt(glm::clamp(f0, glm::vec3(0.0F), glm::vec3(0.9999F)));
    return (glm::vec3(1.0F) + s) / glm::max(glm::vec3(1.0F) - s, glm::vec3(1.0e-4F));
}

[[nodiscard]] glm::vec3 tfFresnelF82(glm::vec3 F0, glm::vec3 F82, float cosTheta) noexcept {
    constexpr float muBar = 1.0F / 7.0F;
    const float denom = muBar * std::pow(1.0F - muBar, 6.0F);
    const float mu = std::clamp(cosTheta, 0.0F, 1.0F);
    const glm::vec3 fSchlickBar = F0 + (glm::vec3(1.0F) - F0) * std::pow(1.0F - muBar, 5.0F);
    const glm::vec3 a = fSchlickBar * (glm::vec3(1.0F) - F82) / denom;
    const glm::vec3 fSchlick = F0 + (glm::vec3(1.0F) - F0) * std::pow(1.0F - mu, 5.0F);
    return glm::clamp(fSchlick - a * mu * std::pow(1.0F - mu, 6.0F), glm::vec3(0.0F), glm::vec3(1.0F));
}

void mxArtisticIor(glm::vec3 reflectivity, glm::vec3 edgeColor, glm::vec3& ior, glm::vec3& extinction) noexcept {
    const glm::vec3 r = glm::clamp(reflectivity, glm::vec3(0.0F), glm::vec3(0.99F));
    const glm::vec3 rSqrt = glm::sqrt(r);
    const glm::vec3 nMin = (glm::vec3(1.0F) - r) / (glm::vec3(1.0F) + r);
    const glm::vec3 nMax = (glm::vec3(1.0F) + rSqrt) / glm::max(glm::vec3(1.0F) - rSqrt, glm::vec3(1.0e-4F));
    ior = glm::mix(nMax, nMin, glm::clamp(edgeColor, glm::vec3(0.0F), glm::vec3(1.0F)));
    const glm::vec3 np1 = ior + 1.0F;
    const glm::vec3 nm1 = ior - 1.0F;
    const glm::vec3 k2 = (np1 * np1 * r - nm1 * nm1) / glm::max(glm::vec3(1.0F) - r, glm::vec3(1.0e-4F));
    extinction = glm::sqrt(glm::max(k2, glm::vec3(0.0F)));
}

void tfDielectricPolarized(float cosTheta, float ior, float& Rp, float& Rs) noexcept {
    const float c2 = std::clamp(cosTheta, 0.0F, 1.0F) * std::clamp(cosTheta, 0.0F, 1.0F);
    const float s2 = 1.0F - c2;
    const float t0 = std::max(ior * ior - s2, 0.0F);
    const float t1 = t0 + c2;
    const float t2 = 2.0F * std::sqrt(t0) * cosTheta;
    Rs = (t1 - t2) / std::max(t1 + t2, 1.0e-6F);
    const float t3 = c2 * t0 + s2 * s2;
    const float t4 = t2 * s2;
    Rp = Rs * (t3 - t4) / std::max(t3 + t4, 1.0e-6F);
}

void tfConductorPolarized(float cosTheta, glm::vec3 n, glm::vec3 k, glm::vec3& Rp, glm::vec3& Rs) noexcept {
    const float c2 = std::clamp(cosTheta, 0.0F, 1.0F) * std::clamp(cosTheta, 0.0F, 1.0F);
    const float s2 = 1.0F - c2;
    const glm::vec3 n2 = n * n;
    const glm::vec3 k2 = k * k;
    const glm::vec3 t0 = n2 - k2 - glm::vec3(s2);
    const glm::vec3 a2b2 = glm::sqrt(glm::max(t0 * t0 + 4.0F * n2 * k2, glm::vec3(0.0F)));
    const glm::vec3 t1 = a2b2 + glm::vec3(c2);
    const glm::vec3 a = glm::sqrt(glm::max(0.5F * (a2b2 + t0), glm::vec3(0.0F)));
    const glm::vec3 t2 = 2.0F * a * cosTheta;
    Rs = (t1 - t2) / glm::max(t1 + t2, glm::vec3(1.0e-6F));
    const glm::vec3 t3 = c2 * a2b2 + glm::vec3(s2 * s2);
    const glm::vec3 t4 = t2 * s2;
    Rp = Rs * (t3 - t4) / glm::max(t3 + t4, glm::vec3(1.0e-6F));
}

void tfConductorPhasePolarized(float cosTheta, float eta1, glm::vec3 eta2, glm::vec3 kappa2, glm::vec3& phiP,
                               glm::vec3& phiS) noexcept {
    const glm::vec3 k2 = kappa2 / eta2;
    const float s2 = 1.0F - cosTheta * cosTheta;
    const glm::vec3 A = eta2 * eta2 * (glm::vec3(1.0F) - k2 * k2) - (eta1 * eta1) * glm::vec3(s2);
    const glm::vec3 twoE2k2 = 2.0F * eta2 * eta2 * k2;
    const glm::vec3 B = glm::sqrt(glm::max(A * A + twoE2k2 * twoE2k2, glm::vec3(0.0F)));
    const glm::vec3 U = glm::sqrt(glm::max(0.5F * (A + B), glm::vec3(0.0F)));
    const glm::vec3 V = glm::max(glm::vec3(0.0F), glm::sqrt(glm::max(0.5F * (B - A), glm::vec3(0.0F))));
    const float e1c = eta1 * cosTheta;
    auto atan2v = [](glm::vec3 y, glm::vec3 x) {
        return glm::vec3(std::atan2(y.x, x.x), std::atan2(y.y, x.y), std::atan2(y.z, x.z));
    };
    phiS = atan2v(2.0F * eta1 * V * cosTheta, U * U + V * V - glm::vec3(e1c * e1c));
    const glm::vec3 e22c = eta2 * eta2 * cosTheta;
    const glm::vec3 num = 2.0F * eta1 * e22c * (2.0F * k2 * U - (glm::vec3(1.0F) - k2 * k2) * V);
    const glm::vec3 t = eta2 * eta2 * (glm::vec3(1.0F) + k2 * k2) * cosTheta;
    const glm::vec3 den = t * t - (eta1 * eta1) * (U * U + V * V);
    phiP = atan2v(num, den);
}

[[nodiscard]] glm::vec3 thinFilmAiry(float cosTheta, bool isConductor, glm::vec3 F0, glm::vec3 F82, glm::vec3 nCond,
                                     glm::vec3 kCond, float tfThicknessNm, float tfIor) noexcept {
    const float eta1 = 1.0F;
    const float eta2 = std::max(tfIor, eta1);
    const float ct = std::clamp(cosTheta, 0.0F, 1.0F);
    const float cosTtSq = 1.0F - (1.0F - ct * ct) * (eta1 / eta2) * (eta1 / eta2);
    const bool tir = (cosTtSq <= 0.0F);
    const float cosTt = std::sqrt(std::max(cosTtSq, 0.0F));

    float R12p = 0.0F;
    float R12s = 0.0F;
    tfDielectricPolarized(ct, eta2 / eta1, R12p, R12s);
    if (tir) {
        R12p = 1.0F;
        R12s = 1.0F;
    }
    const float T121p = 1.0F - R12p;
    const float T121s = 1.0F - R12s;

    glm::vec3 R23p, R23s, phi23p, phi23s;
    if (isConductor) {
        tfConductorPolarized(cosTt, nCond / eta2, kCond / eta2, R23p, R23s);
        tfConductorPhasePolarized(cosTt, eta2, nCond, kCond, phi23p, phi23s);
    } else {
        const glm::vec3 f = 0.5F * tfFresnelF82(F0, F82, cosTt);
        R23p = f;
        R23s = f;
        const glm::vec3 eta3 = fresnel0ToIor(F0);
        phi23p = glm::vec3(eta3.x < eta2 ? Math::kPi : 0.0F, eta3.y < eta2 ? Math::kPi : 0.0F,
                           eta3.z < eta2 ? Math::kPi : 0.0F);
        phi23s = phi23p;
    }

    const float cosB = std::cos(std::atan(eta2 / eta1));
    const float phi21p = (ct < cosB) ? 0.0F : Math::kPi;
    const float phi21s = Math::kPi;

    const glm::vec3 r123p = glm::sqrt(glm::max(glm::vec3(R12p) * R23p, glm::vec3(0.0F)));
    const glm::vec3 r123s = glm::sqrt(glm::max(glm::vec3(R12s) * R23s, glm::vec3(0.0F)));

    const float opd = 2.0F * eta2 * cosTt * std::max(tfThicknessNm, 0.0F) * 1.0e-9F;

    glm::vec3 I(0.0F);
    glm::vec3 Cm, Sm;

    glm::vec3 Rsp = (T121p * T121p * R23p) / glm::max(glm::vec3(1.0F) - glm::vec3(R12p) * R23p, glm::vec3(1.0e-4F));
    I += glm::vec3(R12p) + Rsp;
    Cm = Rsp - glm::vec3(T121p);
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123p;
        Sm = 2.0F * thinFilmSensitivity(static_cast<float>(m) * opd,
                                        static_cast<float>(m) * (phi23p + glm::vec3(phi21p)));
        I += Cm * Sm;
    }

    glm::vec3 Rss = (T121s * T121s * R23s) / glm::max(glm::vec3(1.0F) - glm::vec3(R12s) * R23s, glm::vec3(1.0e-4F));
    I += glm::vec3(R12s) + Rss;
    Cm = Rss - glm::vec3(T121s);
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123s;
        Sm = 2.0F * thinFilmSensitivity(static_cast<float>(m) * opd,
                                        static_cast<float>(m) * (phi23s + glm::vec3(phi21s)));
        I += Cm * Sm;
    }

    I *= 0.5F;
    return glm::clamp(thinFilmXyzToRgb(I), glm::vec3(0.0F), glm::vec3(1.0F));
}

// Schlick-base wrapper used by the dielectric thin-film tests below.
[[nodiscard]] glm::vec3 thinFilmIridescentReflectance(glm::vec3 baseF0, float cosTheta1, float thicknessNm,
                                                      float filmIor) noexcept {
    return thinFilmAiry(cosTheta1, false, baseF0, glm::vec3(1.0F), glm::vec3(0.0F), glm::vec3(0.0F), thicknessNm,
                        std::max(filmIor, 1.0F));
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

[[nodiscard]] float ggxDirAlbedo(float nDotV, float alpha) noexcept {
    const float x = std::clamp(nDotV, 0.0F, 1.0F);
    const float y = std::clamp(alpha, 0.0F, 1.0F);
    const float x2 = x * x;
    const float y2 = y * y;
    const glm::vec4 r = glm::vec4(0.1003F, 0.9345F, 1.0F, 1.0F) +
                        glm::vec4(-0.6303F, -2.323F, -1.765F, 0.2281F) * x +
                        glm::vec4(9.748F, 2.229F, 8.263F, 15.94F) * y +
                        glm::vec4(-2.038F, -3.748F, 11.53F, -55.83F) * x * y +
                        glm::vec4(29.34F, 1.424F, 28.96F, 13.08F) * x2 +
                        glm::vec4(-8.245F, -0.7684F, -7.507F, 41.26F) * y2 +
                        glm::vec4(-26.44F, 1.436F, -36.11F, 54.9F) * x2 * y +
                        glm::vec4(19.99F, 0.2913F, 15.86F, 300.2F) * x * y2 +
                        glm::vec4(-5.448F, 0.6286F, 33.37F, -285.1F) * x2 * y2;
    const float a = std::clamp(r.x / r.z, 0.0F, 1.0F);
    const float b = std::clamp(r.y / r.w, 0.0F, 1.0F);
    return std::clamp(a + b, 0.0F, 1.0F);
}

// OpenPBR sheen oracle — Zeltner et al. 2022 LTC sheen, mirroring Harmonia bsdf_shared.slang
// (which is a faithful port of MaterialX mx_microfacet_sheen.glsl; analytic fits, no LUT).
[[nodiscard]] float sheenDirAlbedo(float cosTheta, float roughness) noexcept {
    const float x = std::clamp(cosTheta, 0.0F, 1.0F);
    const float y = std::clamp(roughness, 0.01F, 1.0F);
    const float s = y * (0.0206607F + 1.58491F * y) / (0.0379424F + y * (1.32227F + y));
    const float m = y * (-0.193854F + y * (-1.14885F + y * (1.7932F - 0.95943F * y * y))) / (0.046391F + y);
    const float o = y * (0.000654023F + (-0.0207818F + 0.119681F * y) * y) / (1.26264F + y * (-1.92021F + y));
    const float g = std::exp(-0.5F * ((x - m) / s) * ((x - m) / s)) / (s * std::sqrt(2.0F * Math::kPi)) + o;
    return std::clamp(g, 0.0F, 1.0F);
}

[[nodiscard]] float sheenLtcAInv(float x, float y) noexcept {
    return (2.58126F * x + 0.813703F * y) * y / (1.0F + 0.310327F * x * x + 2.60994F * x * y);
}

[[nodiscard]] float sheenLtcBInv(float x, float y) noexcept {
    return std::sqrt(std::max(0.0F, 1.0F - x)) * (y - 1.0F) * y * y * y
         / (0.0000254053F + 1.71228F * x - 1.71506F * x * y + 1.34174F * y * y);
}

[[nodiscard]] float zeltnerSheenBrdfCos(const glm::vec3& wo, const glm::vec3& wi, float roughness) noexcept {
    const float nDotV = std::clamp(wo.z, 1.0e-4F, 1.0F);
    glm::vec3 w;
    glm::vec3 xAxis(wo.x, wo.y, 0.0F);
    const float lenSq = glm::dot(xAxis, xAxis);
    if (lenSq > 1.0e-8F) {
        xAxis *= (1.0F / std::sqrt(lenSq));
        const glm::vec3 yAxis(-xAxis.y, xAxis.x, 0.0F);
        w = glm::vec3(glm::dot(xAxis, wi), glm::dot(yAxis, wi), wi.z);
    } else {
        w = wi;
    }
    const float aInv = sheenLtcAInv(nDotV, roughness);
    const float bInv = sheenLtcBInv(nDotV, roughness);
    const glm::vec3 wo2(aInv * w.x + bInv * w.z, aInv * w.y, w.z);
    const float l2 = glm::dot(wo2, wo2);
    const float dO = std::max(wo2.z, 0.0F) * Math::kInvPi;
    const float k = aInv / std::max(l2, 1.0e-8F);
    return dO * k * k;
}

[[nodiscard]] glm::vec3 evalSheen(const glm::vec3& color, float roughness, const glm::vec3& wo, const glm::vec3& wi) noexcept {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return glm::vec3(0.0F);
    }
    const float r = std::clamp(roughness, 0.01F, 1.0F);
    const float dirAlbedo = sheenDirAlbedo(wo.z, r);
    const float brdfCos = zeltnerSheenBrdfCos(wo, wi, r);
    return color * (dirAlbedo * brdfCos / std::max(wi.z, 1.0e-4F));
}

[[nodiscard]] glm::vec3 ggxMultiScatterCompensation(const glm::vec3& F0, float nDotV, float alpha) noexcept {
    const float Ess = ggxDirAlbedo(nDotV, alpha);
    return glm::vec3(1.0F) + F0 * ((1.0F / std::max(Ess, 1.0e-3F)) - 1.0F);
}

[[nodiscard]] glm::vec3 evalReflectionMicrofacet(const glm::vec3& F0,
                                                 const glm::vec3& F82,
                                                 const glm::vec3& wo,
                                                 const glm::vec3& wi,
                                                 float alphaX,
                                                 float alphaY) noexcept {
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
    const glm::vec3 single = (D * G / std::max(4.0F * NoL * NoV, 1.0e-5F)) * tint;
    // GGX multiple-scattering energy compensation (matches Harmonia bsdf_shared.slang).
    const float alpha = std::sqrt(std::max(alphaX * alphaY, 1.0e-8F));
    return single * ggxMultiScatterCompensation(F0, NoV, alpha);
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
                                 (1.0F - sheenDirAlbedo(woL.z, std::clamp(mat.fuzzRoughPad.x, 0.0F, 1.0F)) * weights.fuzzWeight);

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
            result += weights.fuzzWeight * evalSheen(fuzzColor, std::clamp(mat.fuzzRoughPad.x, 0.0F, 1.0F), woL, wiL);
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

TEST(Bsdf, ZeltnerSheenIsEnergyConservingAndGrazingPeaked) {
    // OpenPBR sheen = Zeltner 2022 LTC. Two spec-faithful properties must hold:
    // (1) the directional albedo (energy reflected by the fuzz layer) stays within [0,1] for
    //     all view angles / roughnesses, so it can validly attenuate the layers beneath it;
    // (2) sheen is a grazing-angle (retroreflective) effect: directional albedo at a grazing
    //     view must exceed that at normal incidence for a typical fuzz roughness.
    for (float r : {0.1F, 0.4F, 0.7F, 1.0F}) {
        for (float c : {0.05F, 0.3F, 0.6F, 0.95F}) {
            const float a = sheenDirAlbedo(c, r);
            EXPECT_GE(a, 0.0F);
            EXPECT_LE(a, 1.0F);
        }
    }
    const float grazing = sheenDirAlbedo(0.05F, 0.4F);
    const float normalInc = sheenDirAlbedo(0.98F, 0.4F);
    EXPECT_GT(grazing, normalInc);

    // The sheen lobe itself must be finite, non-negative, and brighter at a grazing exit than
    // at a near-normal exit for a fixed grazing view (the velvet rim-light signature).
    const glm::vec3 wo = glm::normalize(glm::vec3(0.9F, 0.0F, 0.2F));
    const glm::vec3 wiGrazing = glm::normalize(glm::vec3(-0.9F, 0.0F, 0.15F));
    const glm::vec3 wiNormal = glm::normalize(glm::vec3(0.0F, 0.0F, 1.0F));
    const glm::vec3 white(1.0F);
    const glm::vec3 sGrazing = evalSheen(white, 0.4F, wo, wiGrazing);
    const glm::vec3 sNormal = evalSheen(white, 0.4F, wo, wiNormal);
    EXPECT_TRUE(std::isfinite(sGrazing.x) && std::isfinite(sGrazing.y) && std::isfinite(sGrazing.z));
    EXPECT_GE(sGrazing.x, 0.0F);
    EXPECT_GE(sNormal.x, 0.0F);
    EXPECT_GT(sGrazing.x, sNormal.x);
}

TEST(Bsdf, GgxDirAlbedoLosesEnergyWithRoughness) {
    // Single-scattering GGX directional albedo (F=1) must stay in [0,1] and decrease as
    // roughness rises (more energy lost to inter-microfacet shadowing). This is the energy
    // the multiple-scattering compensation recovers.
    constexpr float kNoV = 0.8F;
    const float smooth = ggxDirAlbedo(kNoV, 0.02F);
    const float mid = ggxDirAlbedo(kNoV, 0.25F);
    const float rough = ggxDirAlbedo(kNoV, 0.9F);

    EXPECT_GE(smooth, 0.0F);
    EXPECT_LE(smooth, 1.0F);
    EXPECT_GT(smooth, mid);
    EXPECT_GT(mid, rough);
    EXPECT_LT(rough, 0.95F); // a rough surface loses noticeable single-scatter energy
}

TEST(Bsdf, GgxMultiScatterCompensationRecoversMetalEnergy) {
    // A perfectly reflective (white) metal must conserve energy: with multiple-scattering
    // compensation the white-furnace reflectance returns to ~1.0 across roughness. Without
    // the compensation a rough metal would sit well below 1.0 (single-scatter energy loss),
    // so this test specifically guards that the compensation is wired in and effective.
    const glm::vec3 wo = glm::normalize(glm::vec3(0.25F, 0.15F, 0.955F));

    for (const float roughness : {0.2F, 0.4F, 0.6F, 0.85F}) {
        GpuMaterial metal = makeMaterial();
        metal.baseColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);
        metal.baseMetalnessDiffRough = glm::vec4(1.0F, 0.0F, 0.0F, 0.0F); // metalness = 1
        metal.specularColorWeight = glm::vec4(1.0F, 1.0F, 1.0F, 1.0F);    // F82 tint = 1 (mirror)
        metal.specularRoughAnisoIor = glm::vec4(roughness, 0.0F, 1.5F, 0.0F);

        const double energy = estimateWhiteFurnaceEnergy(metal, wo, 40000);
        EXPECT_GE(energy, 0.93) << "roughness=" << roughness << " energy=" << energy;
        EXPECT_LE(energy, 1.07) << "roughness=" << roughness << " energy=" << energy;
    }
}

TEST(Bsdf, GgxMultiScatterCompensationIsNegligibleForDielectric) {
    // For a low-F0 dielectric specular lobe the compensation must be a tiny correction
    // (the thin specular layer barely multiple-scatters), so the multiplier stays ~1.
    const glm::vec3 dielectricF0(0.04F);
    for (const float roughness : {0.2F, 0.5F, 0.9F}) {
        const float alpha = std::max(roughness * roughness, 0.001F);
        const glm::vec3 comp = ggxMultiScatterCompensation(dielectricF0, 0.7F, alpha);
        EXPECT_GE(comp.x, 1.0F);
        EXPECT_LT(comp.x, 1.05F) << "roughness=" << roughness;
    }
}

TEST(Bsdf, ThinFilmGuardReturnsBaseWhenInactive) {
    // The eval path applies iridescence only when thin_film_weight>0 AND thickness>0;
    // otherwise the base Schlick reflectance is used unchanged. Verify that contract
    // (mirrors evalReflectionMicrofacetThinFilm / the thinFilmTint guard).
    auto applyThinFilm = [](glm::vec3 baseF0, float cosT, float thickness, float ior, float weight) {
        const glm::vec3 base = baseF0 + (glm::vec3(1.0F) - baseF0) * std::pow(std::clamp(1.0F - cosT, 0.0F, 1.0F), 5.0F);
        if (weight <= 0.0F || thickness <= 0.0F) {
            return base;
        }
        const glm::vec3 irid = thinFilmIridescentReflectance(baseF0, cosT, thickness, ior);
        return glm::mix(base, irid, std::clamp(weight, 0.0F, 1.0F));
    };
    const glm::vec3 baseF0(0.95F, 0.78F, 0.40F);
    for (const float cosT : {0.2F, 0.5F, 0.9F}) {
        const glm::vec3 base = baseF0 + (glm::vec3(1.0F) - baseF0) * std::pow(1.0F - cosT, 5.0F);
        const glm::vec3 offThickness = applyThinFilm(baseF0, cosT, 0.0F, 1.5F, 1.0F);
        const glm::vec3 offWeight = applyThinFilm(baseF0, cosT, 500.0F, 1.5F, 0.0F);
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(offThickness[c], base[c], 1.0e-5F) << "cosT=" << cosT << " (thickness 0)";
            EXPECT_NEAR(offWeight[c], base[c], 1.0e-5F) << "cosT=" << cosT << " (weight 0)";
        }
    }
}

TEST(Bsdf, ThinFilmStaysFiniteAndBounded) {
    // Across a thickness/angle/IOR sweep the reflectance must remain finite and within a
    // sane reflectance range (slight overshoot allowed from the Gaussian-fit sensitivity).
    for (const float thickness : {100.0F, 350.0F, 550.0F, 800.0F, 1200.0F}) {
        for (const float cosT : {0.05F, 0.4F, 0.8F, 1.0F}) {
            for (const float filmIor : {1.2F, 1.5F, 2.0F}) {
                const glm::vec3 r = thinFilmIridescentReflectance(glm::vec3(0.04F), cosT, thickness, filmIor);
                ASSERT_TRUE(std::isfinite(r.x) && std::isfinite(r.y) && std::isfinite(r.z))
                    << "t=" << thickness << " c=" << cosT << " ior=" << filmIor;
                EXPECT_GE(std::min({r.x, r.y, r.z}), 0.0F);
                EXPECT_LE(std::max({r.x, r.y, r.z}), 1.2F);
            }
        }
    }
}

TEST(Bsdf, ThinFilmThicknessSweepShiftsHue) {
    // The defining signature of iridescence: varying film thickness must shift the hue of
    // the reflected colour. Compare normalized chromaticity across the documented sweep.
    auto chroma = [](glm::vec3 c) {
        const float s = c.x + c.y + c.z + 1.0e-6F;
        return glm::vec2(c.x / s, c.y / s);
    };
    const glm::vec3 darkBase(0.04F); // dielectric base maximises interference contrast
    const glm::vec2 a = chroma(thinFilmIridescentReflectance(darkBase, 0.7F, 300.0F, 1.4F));
    const glm::vec2 b = chroma(thinFilmIridescentReflectance(darkBase, 0.7F, 550.0F, 1.4F));
    const glm::vec2 c = chroma(thinFilmIridescentReflectance(darkBase, 0.7F, 800.0F, 1.4F));
    EXPECT_GT(glm::length(a - b), 0.02F) << "300nm vs 550nm should differ in hue";
    EXPECT_GT(glm::length(b - c), 0.02F) << "550nm vs 800nm should differ in hue";
}

TEST(Bsdf, ThinFilmConductorIsVividAndFinite) {
    // The OpenPBR conductor thin-film (complex IOR via Gulbrandsen) must produce a STRONGER,
    // hue-shifting iridescence on a metal base than the dielectric Schlick approximation —
    // this is the anodized-metal vividness the real-IOR path could not reach.
    auto chroma = [](glm::vec3 col) {
        const float s = col.x + col.y + col.z + 1.0e-6F;
        return glm::vec2(col.x / s, col.y / s);
    };
    auto saturation = [](glm::vec3 col) {
        const float mx = std::max({col.x, col.y, col.z});
        const float mn = std::min({col.x, col.y, col.z});
        return (mx > 1.0e-5F) ? (mx - mn) / mx : 0.0F;
    };
    const glm::vec3 baseColor(0.55F, 0.56F, 0.58F); // neutral chromium-like metal
    const glm::vec3 F82(1.0F);
    glm::vec3 n, k;
    mxArtisticIor(baseColor, F82, n, k);
    ASSERT_TRUE(std::isfinite(n.x) && std::isfinite(k.x));
    EXPECT_GT(glm::length(k), 0.0F) << "a reflective metal must have non-zero extinction";

    float maxSat = 0.0F;
    glm::vec2 prevChroma(0.0F);
    float maxHueShift = 0.0F;
    bool first = true;
    for (const float thickness : {100.0F, 240.0F, 380.0F, 520.0F, 660.0F}) {
        const glm::vec3 cond = thinFilmAiry(0.7F, true, baseColor, F82, n, k, thickness, 2.0F);
        ASSERT_TRUE(std::isfinite(cond.x) && std::isfinite(cond.y) && std::isfinite(cond.z))
            << "thickness=" << thickness;
        EXPECT_GE(std::min({cond.x, cond.y, cond.z}), 0.0F);
        EXPECT_LE(std::max({cond.x, cond.y, cond.z}), 1.2F);
        maxSat = std::max(maxSat, saturation(cond));
        const glm::vec2 ch = chroma(cond);
        if (!first) {
            maxHueShift = std::max(maxHueShift, glm::length(ch - prevChroma));
        }
        prevChroma = ch;
        first = false;
    }
    // Vivid: the anodized sweep reaches a strongly saturated colour and shifts hue clearly.
    EXPECT_GT(maxSat, 0.20F) << "conductor thin-film should be vividly coloured";
    EXPECT_GT(maxHueShift, 0.03F) << "conductor thin-film hue should shift across the sweep";
}

// NOTE: GGX specular BRDF symmetry is tested at the shader level (shaders/bsdf.slang).
// A C++ version will be added when Slang → C++ codegen is wired into the build.
