#include <slang-math/slang-math.hpp>

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

[[nodiscard]] sm::float3 sampleUniformHemisphere(float u1, float u2) noexcept {
    const float z = u1;
    const float r = std::sqrt(std::max(0.0F, 1.0F - (z * z)));
    const float phi = Math::k2Pi * u2;
    return sm::float3(r * std::cos(phi), r * std::sin(phi), z);
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

[[nodiscard]] float smithG2(sm::float3 l, sm::float3 v, float alpha) noexcept {
    // Height-correlated Smith masking-shadowing (Heitz 2014), matching the shader's
    // GGX_G2 in Harmonia math.slang: 1 / (1 + Lambda(l) + Lambda(v)). GGX_Lambda is sign-symmetric
    // (uses |v.z|), so this is correct for BOTH reflection (l,v same hemisphere) and transmission
    // (l,v opposite hemispheres) — previously max(.,0) zeroed the transmitted leg. Consistent with
    // the MaterialX `mx_ggx_dir_albedo_analytic` fit used for multiple-scattering energy
    // compensation. (Earlier this oracle used the separable G1(l)*G1(v) form, which diverged from
    // the shader at the grazing + high-roughness corner — surfaced by the V0 GGX-albedo test.)
    const float lambdaL = smithLambdaGgx(std::abs(l.z), alpha);
    const float lambdaV = smithLambdaGgx(std::abs(v.z), alpha);
    return 1.0F / (1.0F + lambdaL + lambdaV);
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
    // G3 (spec): specular_weight modulates the dielectric F0/eta (see the modulated eta below), NOT
    // the dielectric lobe weight; for the metal base it is applied directly as the metal BSDF weight.
    // At specular_weight=0 the modulated eta→1 (no interface, zero reflection at all angles); the
    // generalized-Schlick Fresnel keeps a grazing term at F0=0, so gate the dielectric lobe to vanish
    // as specular_weight→0 (a no-op at the default specular_weight=1).
    const float specularParam = std::clamp(mat.specularColorWeight.w, 0.0F, 1.0F);
    const float dielectricPresence = sm::smoothstep(0.0F, 0.03F, specularParam);
    weights.specularWeight = opaqueBase * (1.0F - metalness) * dielectricPresence;
    weights.metalWeight = baseWeight * metalness * specularParam;
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

[[nodiscard]] sm::float3 sampleUniformSphere(float u1, float u2) noexcept {
    const float z = 1.0F - (2.0F * u1);
    const float r = std::sqrt(std::max(0.0F, 1.0F - (z * z)));
    const float phi = Math::k2Pi * u2;
    return sm::float3(r * std::cos(phi), r * std::sin(phi), z);
}

[[nodiscard]] sm::float3 toLocal(const sm::float3& v, const sm::float3& T, const sm::float3& B, const sm::float3& N) noexcept {
    return sm::float3(sm::dot(v, T), sm::dot(v, B), sm::dot(v, N));
}

void orientFrame(const sm::float3& wo, const sm::float3& N, const sm::float3& T, const sm::float3& B,
                 sm::float3& Ns, sm::float3& Ts, sm::float3& Bs) noexcept {
    if (sm::dot(wo, N) >= 0.0F) {
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
    mat.baseColorWeight = sm::float4(1.0F);
    mat.baseMetalnessDiffRough = sm::float4(0.0F);
    // OpenPBR default specular: specular_color = (1,1,1), specular_weight = 1.0. NOTE: the 4th
    // channel (.w) is specular_weight, NOT colour — it must be 1.0, not 0.04. (The dielectric F0
    // ~0.04 comes from specular_ior=1.5 via iorToF0, multiplied by specular_color in evalBSDF.)
    mat.specularColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    mat.specularRoughAnisoIor = sm::float4(0.2F, 0.0F, 1.5F, 0.0F);
    mat.transmissionColorWeight = sm::float4(0.0F);
    mat.transmissionParams = sm::float4(0.0F);
    mat.transmissionScatter = sm::float4(0.0F);
    mat.subsurfaceColorWeight = sm::float4(0.0F);
    mat.subsurfaceRadiusScale = sm::float4(0.0F);
    mat.thinFilmParams = sm::float4(0.0F);
    mat.coatColorWeight = sm::float4(0.0F);
    mat.coatRoughAnisoIorDark = sm::float4(0.0F);
    mat.fuzzColorWeight = sm::float4(0.0F);
    mat.fuzzRoughPad = sm::float4(0.0F);
    mat.opacityFlagsPad = sm::float4(1.0F, 0.0F, 0.0F, 0.0F);
    return mat;
}

[[nodiscard]] sm::float2 computeAlpha(float roughness, float anisotropy) noexcept {
    const float r = std::max(roughness, 0.02F);
    const float a = std::clamp(anisotropy, 0.0F, 0.98F);
    const float alphaX = std::max(r * r * std::sqrt(2.0F / (1.0F + ((1.0F - a) * (1.0F - a)))), 0.001F);
    const float alphaY = std::max((1.0F - a) * alphaX, 0.001F);
    return sm::float2(alphaX, alphaY);
}

[[nodiscard]] sm::float3 iorToF0(float eta) noexcept {
    const float x = (eta - 1.0F) / std::max(eta + 1.0F, 1.0e-4F);
    return sm::float3(x * x);
}

// ─── OpenPBR thin-film Airy (mirror of Harmonia bsdf_shared.slang mx_fresnel_airy) ───
[[nodiscard]] sm::float3 mx_eval_sensitivity(float opd, sm::float3 shift) noexcept {
    const float phase = 2.0F * Math::kPi * opd;
    const sm::float3 val(5.4856e-13F, 4.4201e-13F, 5.2481e-13F);
    const sm::float3 pos(1.6810e+06F, 1.7953e+06F, 2.2084e+06F);
    const sm::float3 var(4.3278e+09F, 9.3046e+09F, 6.6121e+09F);
    sm::float3 xyz = val * sm::sqrt(2.0F * Math::kPi * var) * sm::cos(pos * phase + shift) *
                    sm::exp(-var * phase * phase);
    xyz.x += 9.7470e-14F * std::sqrt(2.0F * Math::kPi * 4.5282e+09F) *
             std::cos(2.2399e+06F * phase + shift.x) * std::exp(-4.5282e+09F * phase * phase);
    return xyz / 1.0685e-7F;
}

[[nodiscard]] sm::float3 mx_xyz_to_rgb(sm::float3 v) noexcept {
    return sm::float3(sm::dot(sm::float3(2.3706743F, -0.9000405F, -0.4706338F), v),
                     sm::dot(sm::float3(-0.5138850F, 1.4253036F, 0.0885814F), v),
                     sm::dot(sm::float3(0.0052982F, -0.0146949F, 1.0093968F), v));
}

[[nodiscard]] sm::float3 mx_f0_to_ior(sm::float3 f0) noexcept {
    const sm::float3 s = sm::sqrt(sm::clamp(f0, sm::float3(0.0F), sm::float3(0.9999F)));
    return (sm::float3(1.0F) + s) / sm::max(sm::float3(1.0F) - s, sm::float3(1.0e-4F));
}

[[nodiscard]] sm::float3 mx_fresnel_F82(sm::float3 F0, sm::float3 F82, float cosTheta) noexcept {
    constexpr float muBar = 1.0F / 7.0F;
    const float denom = muBar * std::pow(1.0F - muBar, 6.0F);
    const float mu = std::clamp(cosTheta, 0.0F, 1.0F);
    const sm::float3 fSchlickBar = F0 + (sm::float3(1.0F) - F0) * std::pow(1.0F - muBar, 5.0F);
    const sm::float3 a = fSchlickBar * (sm::float3(1.0F) - F82) / denom;
    const sm::float3 fSchlick = F0 + (sm::float3(1.0F) - F0) * std::pow(1.0F - mu, 5.0F);
    return sm::clamp(fSchlick - a * mu * std::pow(1.0F - mu, 6.0F), sm::float3(0.0F), sm::float3(1.0F));
}

void mx_artistic_ior(sm::float3 reflectivity, sm::float3 edgeColor, sm::float3& ior, sm::float3& extinction) noexcept {
    const sm::float3 r = sm::clamp(reflectivity, sm::float3(0.0F), sm::float3(0.99F));
    const sm::float3 rSqrt = sm::sqrt(r);
    const sm::float3 nMin = (sm::float3(1.0F) - r) / (sm::float3(1.0F) + r);
    const sm::float3 nMax = (sm::float3(1.0F) + rSqrt) / sm::max(sm::float3(1.0F) - rSqrt, sm::float3(1.0e-4F));
    ior = sm::lerp(nMax, nMin, sm::clamp(edgeColor, sm::float3(0.0F), sm::float3(1.0F)));
    const sm::float3 np1 = ior + 1.0F;
    const sm::float3 nm1 = ior - 1.0F;
    const sm::float3 k2 = (np1 * np1 * r - nm1 * nm1) / sm::max(sm::float3(1.0F) - r, sm::float3(1.0e-4F));
    extinction = sm::sqrt(sm::max(k2, sm::float3(0.0F)));
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

void tfConductorPolarized(float cosTheta, sm::float3 n, sm::float3 k, sm::float3& Rp, sm::float3& Rs) noexcept {
    const float c2 = std::clamp(cosTheta, 0.0F, 1.0F) * std::clamp(cosTheta, 0.0F, 1.0F);
    const float s2 = 1.0F - c2;
    const sm::float3 n2 = n * n;
    const sm::float3 k2 = k * k;
    const sm::float3 t0 = n2 - k2 - sm::float3(s2);
    const sm::float3 a2b2 = sm::sqrt(sm::max(t0 * t0 + 4.0F * n2 * k2, sm::float3(0.0F)));
    const sm::float3 t1 = a2b2 + sm::float3(c2);
    const sm::float3 a = sm::sqrt(sm::max(0.5F * (a2b2 + t0), sm::float3(0.0F)));
    const sm::float3 t2 = 2.0F * a * cosTheta;
    Rs = (t1 - t2) / sm::max(t1 + t2, sm::float3(1.0e-6F));
    const sm::float3 t3 = c2 * a2b2 + sm::float3(s2 * s2);
    const sm::float3 t4 = t2 * s2;
    Rp = Rs * (t3 - t4) / sm::max(t3 + t4, sm::float3(1.0e-6F));
}

void tfConductorPhasePolarized(float cosTheta, float eta1, sm::float3 eta2, sm::float3 kappa2, sm::float3& phiP,
                               sm::float3& phiS) noexcept {
    const sm::float3 k2 = kappa2 / eta2;
    const float s2 = 1.0F - cosTheta * cosTheta;
    const sm::float3 A = eta2 * eta2 * (sm::float3(1.0F) - k2 * k2) - (eta1 * eta1) * sm::float3(s2);
    const sm::float3 twoE2k2 = 2.0F * eta2 * eta2 * k2;
    const sm::float3 B = sm::sqrt(sm::max(A * A + twoE2k2 * twoE2k2, sm::float3(0.0F)));
    const sm::float3 U = sm::sqrt(sm::max(0.5F * (A + B), sm::float3(0.0F)));
    const sm::float3 V = sm::max(sm::float3(0.0F), sm::sqrt(sm::max(0.5F * (B - A), sm::float3(0.0F))));
    const float e1c = eta1 * cosTheta;
    auto atan2v = [](sm::float3 y, sm::float3 x) {
        return sm::float3(std::atan2(y.x, x.x), std::atan2(y.y, x.y), std::atan2(y.z, x.z));
    };
    phiS = atan2v(2.0F * eta1 * V * cosTheta, U * U + V * V - sm::float3(e1c * e1c));
    const sm::float3 e22c = eta2 * eta2 * cosTheta;
    const sm::float3 num = 2.0F * eta1 * e22c * (2.0F * k2 * U - (sm::float3(1.0F) - k2 * k2) * V);
    const sm::float3 t = eta2 * eta2 * (sm::float3(1.0F) + k2 * k2) * cosTheta;
    const sm::float3 den = t * t - (eta1 * eta1) * (U * U + V * V);
    phiP = atan2v(num, den);
}

[[nodiscard]] sm::float3 mx_fresnel_airy(float cosTheta, bool isConductor, sm::float3 F0, sm::float3 F82, sm::float3 nCond,
                                     sm::float3 kCond, float tfThicknessNm, float tfIor) noexcept {
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

    sm::float3 R23p, R23s, phi23p, phi23s;
    if (isConductor) {
        tfConductorPolarized(cosTt, nCond / eta2, kCond / eta2, R23p, R23s);
        tfConductorPhasePolarized(cosTt, eta2, nCond, kCond, phi23p, phi23s);
    } else {
        const sm::float3 f = 0.5F * mx_fresnel_F82(F0, F82, cosTt);
        R23p = f;
        R23s = f;
        const sm::float3 eta3 = mx_f0_to_ior(F0);
        phi23p = sm::float3(eta3.x < eta2 ? Math::kPi : 0.0F, eta3.y < eta2 ? Math::kPi : 0.0F,
                           eta3.z < eta2 ? Math::kPi : 0.0F);
        phi23s = phi23p;
    }

    const float cosB = std::cos(std::atan(eta2 / eta1));
    const float phi21p = (ct < cosB) ? 0.0F : Math::kPi;
    const float phi21s = Math::kPi;

    const sm::float3 r123p = sm::sqrt(sm::max(sm::float3(R12p) * R23p, sm::float3(0.0F)));
    const sm::float3 r123s = sm::sqrt(sm::max(sm::float3(R12s) * R23s, sm::float3(0.0F)));

    const float opd = 2.0F * eta2 * cosTt * std::max(tfThicknessNm, 0.0F) * 1.0e-9F;

    sm::float3 I(0.0F);
    sm::float3 Cm, Sm;

    sm::float3 Rsp = (T121p * T121p * R23p) / sm::max(sm::float3(1.0F) - sm::float3(R12p) * R23p, sm::float3(1.0e-4F));
    I += sm::float3(R12p) + Rsp;
    Cm = Rsp - sm::float3(T121p);
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123p;
        Sm = 2.0F * mx_eval_sensitivity(static_cast<float>(m) * opd,
                                        static_cast<float>(m) * (phi23p + sm::float3(phi21p)));
        I += Cm * Sm;
    }

    sm::float3 Rss = (T121s * T121s * R23s) / sm::max(sm::float3(1.0F) - sm::float3(R12s) * R23s, sm::float3(1.0e-4F));
    I += sm::float3(R12s) + Rss;
    Cm = Rss - sm::float3(T121s);
    for (int m = 1; m <= 2; ++m) {
        Cm *= r123s;
        Sm = 2.0F * mx_eval_sensitivity(static_cast<float>(m) * opd,
                                        static_cast<float>(m) * (phi23s + sm::float3(phi21s)));
        I += Cm * Sm;
    }

    I *= 0.5F;
    return sm::clamp(mx_xyz_to_rgb(I), sm::float3(0.0F), sm::float3(1.0F));
}

// Schlick-base wrapper used by the dielectric thin-film tests below.
[[nodiscard]] sm::float3 thinFilmIridescentReflectance(sm::float3 baseF0, float cosTheta1, float thicknessNm,
                                                      float filmIor) noexcept {
    return mx_fresnel_airy(cosTheta1, false, baseF0, sm::float3(1.0F), sm::float3(0.0F), sm::float3(0.0F), thicknessNm,
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

[[nodiscard]] sm::float3 evalDiffuse(const sm::float3& color, float roughness, const sm::float3& wo, const sm::float3& wi) noexcept {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return sm::float3(0.0F);
    }

    const float NdotV = wo.z;
    const float NdotL = wi.z;
    const float s = sm::dot(wi, wo) - (NdotL * NdotV);
    const float stinv = (s > 0.0F) ? (s / std::max(NdotL, NdotV)) : s;

    const float A = 1.0F / (1.0F + (0.146975F * roughness));
    const sm::float3 lobeSingle = color * A * (1.0F + (roughness * stinv));

    const float dAlbedoV = diffuseDirAlbedoFujii(NdotV, roughness);
    const float dAlbedoL = diffuseDirAlbedoFujii(NdotL, roughness);
    const float avgA = diffuseAvgAlbedoFujii(roughness);
    const sm::float3 colorMS = (color * color * avgA) / sm::max(sm::float3(0.0001F), sm::float3(1.0F) - color * std::max(0.0F, 1.0F - avgA));
    const sm::float3 lobeMS = colorMS * std::max(1.0e-4F, 1.0F - dAlbedoV) * std::max(1.0e-4F, 1.0F - dAlbedoL) /
                             std::max(1.0e-4F, 1.0F - avgA);

    return (lobeSingle + lobeMS) * Math::kInvPi;
}

[[nodiscard]] sm::float2 mx_ggx_dir_albedo_ab(float nDotV, float alpha) noexcept {
    const float x = std::clamp(nDotV, 0.0F, 1.0F);
    const float y = std::clamp(alpha, 0.0F, 1.0F);
    const float x2 = x * x;
    const float y2 = y * y;
    const sm::float4 r = sm::float4(0.1003F, 0.9345F, 1.0F, 1.0F) +
                        sm::float4(-0.6303F, -2.323F, -1.765F, 0.2281F) * x +
                        sm::float4(9.748F, 2.229F, 8.263F, 15.94F) * y +
                        sm::float4(-2.038F, -3.748F, 11.53F, -55.83F) * x * y +
                        sm::float4(29.34F, 1.424F, 28.96F, 13.08F) * x2 +
                        sm::float4(-8.245F, -0.7684F, -7.507F, 41.26F) * y2 +
                        sm::float4(-26.44F, 1.436F, -36.11F, 54.9F) * x2 * y +
                        sm::float4(19.99F, 0.2913F, 15.86F, 300.2F) * x * y2 +
                        sm::float4(-5.448F, 0.6286F, 33.37F, -285.1F) * x2 * y2;
    return sm::float2(std::clamp(r.x / r.z, 0.0F, 1.0F), std::clamp(r.y / r.w, 0.0F, 1.0F));
}

[[nodiscard]] float mx_ggx_dir_albedo(float nDotV, float alpha) noexcept {
    const sm::float2 ab = mx_ggx_dir_albedo_ab(nDotV, alpha);
    return std::clamp(ab.x + ab.y, 0.0F, 1.0F);
}

// Fresnel-weighted GGX directional albedo (mirrors Harmonia bsdf_shared.slang
// `fresnelGgxDirAlbedo`): F0*A + F90*B with F90 = 1 (MaterialX mx_ggx_dir_albedo). The previous
// Schlick form ignored alpha; this honors roughness via the rational fit.
[[nodiscard]] float fresnelGgxDirAlbedo(float nDotV, float f0, float alpha) noexcept {
    const sm::float2 ab = mx_ggx_dir_albedo_ab(nDotV, alpha);
    return std::clamp(f0 * ab.x + ab.y, 0.0F, 1.0F);
}

// OpenPBR sheen oracle — Zeltner et al. 2022 LTC sheen, mirroring Harmonia bsdf_shared.slang
// (which is a faithful port of MaterialX mx_microfacet_sheen.glsl; analytic fits, no LUT).
[[nodiscard]] float mx_zeltner_sheen_dir_albedo(float cosTheta, float roughness) noexcept {
    const float x = std::clamp(cosTheta, 0.0F, 1.0F);
    const float y = std::clamp(roughness, 0.01F, 1.0F);
    const float s = y * (0.0206607F + 1.58491F * y) / (0.0379424F + y * (1.32227F + y));
    const float m = y * (-0.193854F + y * (-1.14885F + y * (1.7932F - 0.95943F * y * y))) / (0.046391F + y);
    const float o = y * (0.000654023F + (-0.0207818F + 0.119681F * y) * y) / (1.26264F + y * (-1.92021F + y));
    const float g = std::exp(-0.5F * ((x - m) / s) * ((x - m) / s)) / (s * std::sqrt(2.0F * Math::kPi)) + o;
    return std::clamp(g, 0.0F, 1.0F);
}

[[nodiscard]] float mx_zeltner_sheen_ltc_aInv(float x, float y) noexcept {
    return (2.58126F * x + 0.813703F * y) * y / (1.0F + 0.310327F * x * x + 2.60994F * x * y);
}

[[nodiscard]] float mx_zeltner_sheen_ltc_bInv(float x, float y) noexcept {
    return std::sqrt(std::max(0.0F, 1.0F - x)) * (y - 1.0F) * y * y * y
         / (0.0000254053F + 1.71228F * x - 1.71506F * x * y + 1.34174F * y * y);
}

[[nodiscard]] float mx_zeltner_sheen_brdf(const sm::float3& wo, const sm::float3& wi, float roughness) noexcept {
    const float nDotV = std::clamp(wo.z, 1.0e-4F, 1.0F);
    sm::float3 w;
    sm::float3 xAxis(wo.x, wo.y, 0.0F);
    const float lenSq = sm::dot(xAxis, xAxis);
    if (lenSq > 1.0e-8F) {
        xAxis *= (1.0F / std::sqrt(lenSq));
        const sm::float3 yAxis(-xAxis.y, xAxis.x, 0.0F);
        w = sm::float3(sm::dot(xAxis, wi), sm::dot(yAxis, wi), wi.z);
    } else {
        w = wi;
    }
    const float aInv = mx_zeltner_sheen_ltc_aInv(nDotV, roughness);
    const float bInv = mx_zeltner_sheen_ltc_bInv(nDotV, roughness);
    const sm::float3 wo2(aInv * w.x + bInv * w.z, aInv * w.y, w.z);
    const float l2 = sm::dot(wo2, wo2);
    const float dO = std::max(wo2.z, 0.0F) * Math::kInvPi;
    const float k = aInv / std::max(l2, 1.0e-8F);
    return dO * k * k;
}

[[nodiscard]] sm::float3 evalSheen(const sm::float3& color, float roughness, const sm::float3& wo, const sm::float3& wi) noexcept {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return sm::float3(0.0F);
    }
    const float r = std::clamp(roughness, 0.01F, 1.0F);
    const float dirAlbedo = mx_zeltner_sheen_dir_albedo(wo.z, r);
    const float brdfCos = mx_zeltner_sheen_brdf(wo, wi, r);
    return color * (dirAlbedo * brdfCos / std::max(wi.z, 1.0e-4F));
}

[[nodiscard]] sm::float3 ggxMultiScatterCompensation(const sm::float3& F0, float nDotV, float alpha) noexcept {
    const float Ess = mx_ggx_dir_albedo(nDotV, alpha);
    // MaterialX mx_ggx_energy_compensation: 1 + Fss*(1-Ess)/Ess, Fss = F0 + (F90-F0)/21, F90=1.
    const sm::float3 Fss = F0 + (sm::float3(1.0F) - F0) * (1.0F / 21.0F);
    return sm::float3(1.0F) + Fss * (1.0F - Ess) / std::max(Ess, 1.0e-3F);
}

[[nodiscard]] sm::float3 evalReflectionMicrofacet(const sm::float3& F0,
                                                 const sm::float3& F82,
                                                 const sm::float3& wo,
                                                 const sm::float3& wi,
                                                 float alphaX,
                                                 float alphaY) noexcept {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return sm::float3(0.0F);
    }

    const sm::float3 h = sm::normalize(wo + wi);
    const float NoH = std::max(h.z, 0.0F);
    const float NoL = std::max(wi.z, 0.0F);
    const float NoV = std::max(wo.z, 0.0F);
    const float VoH = std::max(sm::dot(wo, h), 0.0F);
    if (NoH <= 0.0F || NoL <= 0.0F || NoV <= 0.0F || VoH <= 0.0F) {
        return sm::float3(0.0F);
    }

    const float D = ggxD(NoH, alphaX);
    const float G = smithG2(wi, wo, alphaX);
    // Generalized-Schlick F82 reflectance (matches the shader's mx_fresnel_F82). The earlier inline
    // `mix(F0, F82, 1-F)` was a WRONG mirror: with specular_color (F82) = 1.0 it returned ~0.96
    // reflectance even at normal incidence, turning every full-specular dielectric into a near
    // mirror. (Masked until now because makeMaterial defaulted specular_weight to 0.04.)
    const sm::float3 F = mx_fresnel_F82(F0, F82, VoH);
    const sm::float3 single = (D * G / std::max(4.0F * NoL * NoV, 1.0e-5F)) * F;
    // GGX multiple-scattering energy compensation (matches Harmonia bsdf_shared.slang).
    const float alpha = std::sqrt(std::max(alphaX * alphaY, 1.0e-8F));
    return single * ggxMultiScatterCompensation(F0, NoV, alpha);
}

[[nodiscard]] sm::float3 evalTransmissionMicrofacet(const sm::float3& color,
                                                   float depth,
                                                   float eta,
                                                   float alphaX,
                                                   float alphaY,
                                                   const sm::float3& wo,
                                                   const sm::float3& wi) noexcept {
    // PBRT-v4 DielectricBxDF rough transmission (Radiance transport). Mirrors Harmonia
    // bsdf_shared.slang evalTransmissionMicrofacet.
    if (wo.z * wi.z >= 0.0F) {
        return sm::float3(0.0F);
    }
    const float cosTheta_o = wo.z;
    const float cosTheta_i = wi.z;
    const float etap = cosTheta_o > 0.0F ? eta : (1.0F / eta);
    sm::float3 wm = wi * etap + wo;
    if (sm::dot(wm, wm) < 1.0e-12F) {
        return sm::float3(0.0F);
    }
    wm = sm::normalize(wm);
    if (wm.z < 0.0F) {
        wm = -wm;
    }
    if (sm::dot(wm, wi) * cosTheta_i < 0.0F || sm::dot(wm, wo) * cosTheta_o < 0.0F) {
        return sm::float3(0.0F);
    }

    const float NoH = std::max(wm.z, 0.0F);
    const float D = ggxD(NoH, alphaX);
    const float G = smithG2(wi, wo, alphaX);
    const float F = fresnelDielectric(std::abs(sm::dot(wo, wm)), eta);
    float denom = sm::dot(wi, wm) + sm::dot(wo, wm) / etap;
    denom = denom * denom * std::abs(cosTheta_i) * std::abs(cosTheta_o);
    float ft = D * (1.0F - F) * G * std::abs(sm::dot(wi, wm) * sm::dot(wo, wm)) / std::max(denom, 1.0e-12F);
    ft /= (etap * etap);   // Radiance transport (camera paths)

    const float msAlpha = std::sqrt(std::max(alphaX * alphaY, 1.0e-8F));
    const float msComp = 1.0F / std::max(mx_ggx_dir_albedo(std::abs(wo.z), msAlpha), 1.0e-3F);
    const sm::float3 absorption = sm::exp(-std::max(depth, 0.001F) * (sm::float3(1.0F) - sm::clamp(color, sm::float3(0.0F), sm::float3(1.0F))));
    return absorption * color * (ft * msComp);
}

// Solid-angle pdf of the refracted ray (PBRT-v4 DielectricBxDF::PDF). Mirrors Harmonia
// bsdf_shared.slang transmissionPdf.
[[nodiscard]] float transmissionPdf(float eta, float alphaX, float alphaY,
                                    const sm::float3& wo, const sm::float3& wi) noexcept {
    if (wo.z * wi.z >= 0.0F) {
        return 0.0F;
    }
    const float cosTheta_o = wo.z;
    const float cosTheta_i = wi.z;
    const float etap = cosTheta_o > 0.0F ? eta : (1.0F / eta);
    sm::float3 wm = wi * etap + wo;
    if (sm::dot(wm, wm) < 1.0e-12F) {
        return 0.0F;
    }
    wm = sm::normalize(wm);
    if (wm.z < 0.0F) {
        wm = -wm;
    }
    if (sm::dot(wm, wi) * cosTheta_i < 0.0F || sm::dot(wm, wo) * cosTheta_o < 0.0F) {
        return 0.0F;
    }
    const float alpha = std::sqrt(std::max(alphaX * alphaY, 1.0e-8F));
    const float D = ggxD(std::max(wm.z, 0.0F), alpha);
    const float G1wo = smithG1(std::max(wo.z, 0.0F), alpha);
    const float pdf_h = G1wo * D * std::abs(sm::dot(wo, wm)) / std::max(std::abs(wo.z), 1.0e-5F);
    const float denom = sm::dot(wi, wm) + sm::dot(wo, wm) / etap;
    const float dwm_dwi = std::abs(sm::dot(wi, wm)) / std::max(denom * denom, 1.0e-12F);
    return pdf_h * dwm_dwi;
}

[[nodiscard]] sm::float3 evalBSDF(const GpuMaterial& mat,
                                 const sm::float3& wo,
                                 const sm::float3& wi,
                                 const sm::float3& N,
                                 const sm::float3& T,
                                 const sm::float3& B,
                                 const sm::float3& cN,
                                 const sm::float3& cT,
                                 const sm::float3& cB) noexcept {
    (void)cN;
    (void)cT;
    (void)cB;
    sm::float3 Ns;
    sm::float3 Ts;
    sm::float3 Bs;
    orientFrame(wo, N, T, B, Ns, Ts, Bs);
    const sm::float3 woL = toLocal(wo, Ts, Bs, Ns);
    const sm::float3 wiL = toLocal(wi, Ts, Bs, Ns);
    if (woL.z == 0.0F || wiL.z == 0.0F) {
        return sm::float3(0.0F);
    }

    const float opacity = std::clamp(mat.opacityFlagsPad.x, 0.0F, 1.0F);
    const sm::float3 baseColor = sm::clamp(sm::float3(mat.baseColorWeight), sm::float3(0.0F), sm::float3(1.0F));
    const sm::float3 specColor = sm::clamp(sm::float3(mat.specularColorWeight), sm::float3(0.0F), sm::float3(1.0F));
    const sm::float3 transColor = sm::clamp(sm::float3(mat.transmissionColorWeight), sm::float3(0.0F), sm::float3(1.0F));
    const sm::float3 subsurfaceColor = sm::clamp(sm::float3(mat.subsurfaceColorWeight), sm::float3(0.0F), sm::float3(1.0F));
    const sm::float3 coatColor = sm::clamp(sm::float3(mat.coatColorWeight), sm::float3(0.0F), sm::float3(1.0F));
    const sm::float3 fuzzColor = sm::clamp(sm::float3(mat.fuzzColorWeight), sm::float3(0.0F), sm::float3(1.0F));

    const float baseRough = mat.specularRoughAnisoIor.x;
    const float coatRough = mat.coatRoughAnisoIorDark.x;
    const float coatEta = std::max(mat.coatRoughAnisoIorDark.z, 1.01F);
    // G3 (mirror of openpbrModulatedEta): the base sees specular_ior/coat_ior (TIR-safe inverted)
    // through the coat, then specular_weight scales F0 (scaled_F0 = specular_weight·F0) and the
    // modulated eta is re-derived from the sign-preserving sqrt of the clamped scaled F0.
    const float specIorRaw = std::max(mat.specularRoughAnisoIor.z, 1.01F);
    float etaRatioG3 = specIorRaw / std::max(coatEta, 1.0e-4F);
    if (etaRatioG3 < 1.0F) {
        etaRatioG3 = coatEta / std::max(specIorRaw, 1.0e-4F);
    }
    const float etaS = std::lerp(specIorRaw, etaRatioG3, std::clamp(mat.coatColorWeight.w, 0.0F, 1.0F));
    const float f0sqrtG3 = (etaS - 1.0F) / (etaS + 1.0F);
    const float scaledF0G3 = std::clamp(std::clamp(mat.specularColorWeight.w, 0.0F, 1.0F) * f0sqrtG3 * f0sqrtG3, 0.0F, 0.99999F);
    const float epsG3 = (etaS >= 1.0F ? 1.0F : -1.0F) * std::sqrt(scaledF0G3);
    const float eta = (1.0F + epsG3) / std::max(1.0F - epsG3, 1.0e-4F);
    const float diffuseRough = std::clamp(mat.baseMetalnessDiffRough.y, 0.0F, 1.0F);
    const float coatDark = std::clamp(mat.coatRoughAnisoIorDark.w, 0.0F, 1.0F);

    // G4 (mirror of bsdf_shared.slang openpbrEffectiveSpecularRoughness): the coat roughens the
    // specular base in roughness space, then computeAlpha — NOT an alpha-space combine.
    const float coatWeightG4 = std::clamp(mat.coatColorWeight.w, 0.0F, 1.0F);
    const float effSpecRough = std::lerp(baseRough,
        std::pow(std::min(1.0F, 2.0F * std::pow(coatRough, 4.0F) + std::pow(baseRough, 4.0F)), 0.25F),
        coatWeightG4);
    const sm::float2 alpha = computeAlpha(effSpecRough, 0.0F);
    const sm::float2 coatAlpha = computeAlpha(coatRough, 0.0F);
    const float alphaX = alpha.x;
    const float alphaY = alpha.y;

    LobeWeights weights = computeLobeWeights(mat);
    // G1 coat darkening (spec base_darkening) + G2 coat_color absorption — per-channel,
    // view-independent multiply. Mirrors bsdf_shared.slang openpbrCoatDarkening +
    // openpbrCoatColorAttenuation.
    const float coatF0v = std::pow((coatEta - 1.0F) / (coatEta + 1.0F), 2.0F);
    const float Kcoat = 1.0F - (1.0F - coatF0v) / std::max(coatEta * coatEta, 1.0e-4F);
    const sm::float3 Emetal = baseColor * std::clamp(mat.specularColorWeight.w, 0.0F, 1.0F);
    const sm::float3 Edielectric = sm::lerp(baseColor, subsurfaceColor, std::clamp(mat.subsurfaceColorWeight.w, 0.0F, 1.0F));
    const sm::float3 Ebase = sm::clamp(sm::lerp(Edielectric, Emetal, std::clamp(mat.baseMetalnessDiffRough.x, 0.0F, 1.0F)), sm::float3(0.0F), sm::float3(1.0F));
    const sm::float3 darkening = sm::lerp(sm::float3(1.0F),
        sm::float3(1.0F - Kcoat) / sm::max(sm::float3(1.0F) - Ebase * Kcoat, sm::float3(1.0e-4F)),
        std::clamp(weights.coatWeight * coatDark, 0.0F, 1.0F));
    const sm::float3 coatAtten = sm::lerp(sm::float3(1.0F), coatColor, std::clamp(weights.coatWeight, 0.0F, 1.0F));
    // G8: fuzz (topmost) attenuates both the base and the coat lobe below it.
    const float fuzzAtten = 1.0F - mx_zeltner_sheen_dir_albedo(woL.z, std::clamp(mat.fuzzRoughPad.x, 0.0F, 1.0F)) * weights.fuzzWeight;
    const sm::float3 baseLayerScale = darkening * coatAtten * fuzzAtten;

    sm::float3 result(0.0F);
    if (wiL.z > 0.0F && woL.z > 0.0F) {
        const sm::float3 dielectricF0 = iorToF0(eta);
        sm::float3 glossyF0 = sm::lerp(dielectricF0 * specColor, baseColor, std::clamp(mat.baseMetalnessDiffRough.x, 0.0F, 1.0F));
        const sm::float3 glossyF82 = specColor;
        const float specF0 = Math::luminance(dielectricF0 * specColor);
        auto underSpec = [&](float c) { return std::clamp(1.0F - std::clamp(specF0 + (1.0F - specF0) * std::pow(1.0F - std::clamp(c, 0.0F, 1.0F), 5.0F), 0.0F, 1.0F), 0.0F, 1.0F); };
        const float diffuseUnderSpec = underSpec(woL.z) * underSpec(wiL.z);

        if (weights.diffuseWeight > 0.0F) {
            result += baseLayerScale * diffuseUnderSpec * weights.diffuseWeight * evalDiffuse(baseColor, diffuseRough, woL, wiL);
        }
        if (weights.subsurfaceWeight > 0.0F && mat.opacityFlagsPad.w > 0.5F) {
            // Thin-walled subsurface only (mirror of bsdf.slang): bulk subsurface is a delta
            // medium-entry handled by the volumetric random walk on the sample side, with no
            // local BRDF value here.
            const float ssScale = std::max(mat.subsurfaceRadiusScale.w, 0.001F);
            const sm::float3 ssTint = subsurfaceColor * (1.0F / (1.0F + ssScale * (mat.subsurfaceRadiusScale.x +
                                                                                 mat.subsurfaceRadiusScale.y +
                                                                                 mat.subsurfaceRadiusScale.z)));
            result += baseLayerScale * diffuseUnderSpec * weights.subsurfaceWeight * evalDiffuse(ssTint, std::clamp(0.25F + 0.5F * diffuseRough, 0.0F, 1.0F), woL, wiL);
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
        const sm::float3 cNs = cN;
        const sm::float3 cTs = cT;
        const sm::float3 cBs = cB;
        const sm::float3 woC = toLocal(wo, cTs, cBs, cNs);
        const sm::float3 wiC = toLocal(wi, cTs, cBs, cNs);
        if (woC.z > 0.0F && wiC.z > 0.0F) {
            const sm::float3 coatF0 = iorToF0(coatEta) * sm::max(coatColor, sm::float3(0.04F));
            result += fuzzAtten * weights.coatWeight * evalReflectionMicrofacet(coatF0, coatColor, woC, wiC, coatAlpha.x, coatAlpha.y);
        }
    }

    if (weights.transmissionWeight > 0.0F) {
        const float transAlphaX = alphaX;
        const float transAlphaY = alphaY;
        result += weights.transmissionWeight *
                  evalTransmissionMicrofacet(transColor, mat.transmissionParams.x, eta, transAlphaX, transAlphaY, woL, wiL);
    }

    if (weights.diffuseTransWeight > 0.0F && woL.z > 0.0F && wiL.z < 0.0F) {
        const sm::float3 wiR(wiL.x, wiL.y, -wiL.z);
        result += weights.diffuseTransWeight * evalDiffuse(subsurfaceColor, diffuseRough, woL, wiR);
    }

    (void)opacity;
    return sm::max(result, sm::float3(0.0F));
}

[[nodiscard]] double estimateWhiteFurnaceEnergy(const GpuMaterial& mat, const sm::float3& wo, int sampleCount) {
    std::mt19937 rng(12345U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const sm::float3 N(0.0F, 0.0F, 1.0F);
    const sm::float3 T(1.0F, 0.0F, 0.0F);
    const sm::float3 B(0.0F, 1.0F, 0.0F);

    double sum = 0.0;
    for (int i = 0; i < sampleCount; ++i) {
        const sm::float3 wi = sampleUniformSphere(dist(rng), dist(rng));
        const sm::float3 f = evalBSDF(mat, wo, wi, N, T, B, N, T, B);
        sum += static_cast<double>(Math::luminance(f) * std::abs(wi.z) * (4.0 * Math::kPi));
    }
    return sum / static_cast<double>(sampleCount);
}

[[nodiscard]] float orenNayarDiffuse(sm::float3 wo, sm::float3 wi, float roughness, float albedo) noexcept {
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
        const sm::float2 wiTangent = sm::normalize(sm::float2(wi.x, wi.y));
        const sm::float2 woTangent = sm::normalize(sm::float2(wo.x, wo.y));
        maxCos = std::max(sm::dot(wiTangent, woTangent), 0.0F);
    }

    const bool incidentIsSteeper = wi.z < wo.z;
    const float sinAlpha = incidentIsSteeper ? sinThetaO : sinThetaI;
    const float tanBeta =
        incidentIsSteeper ? (sinThetaI / std::max(wi.z, kEpsilon)) : (sinThetaO / std::max(wo.z, kEpsilon));
    return (albedo * Math::kInvPi) * (a + (b * maxCos * sinAlpha * tanBeta));
}

// ── V0 OpenPBR numeric-conformance helpers ───────────────────────────────────────────────
// Hyperion's BSDF is the assumed ground truth; OpenPBR's reference implementation is MaterialX
// (`mx_*` genGLSL). These helpers let us validate the analytic MaterialX fits we ported (e.g.
// `mx_ggx_dir_albedo` = `mx_ggx_dir_albedo_analytic`) against ground-truth Monte-Carlo integration
// of the actual BRDFs, and to check physical properties (reciprocity, energy normalization).

// Heitz 2018 "Sampling the GGX Distribution of Visible Normals" (isotropic).
[[nodiscard]] sm::float3 sampleGgxVndf(const sm::float3& Ve, float alpha, float u1, float u2) noexcept {
    const sm::float3 Vh = sm::normalize(sm::float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    const float lensq = (Vh.x * Vh.x) + (Vh.y * Vh.y);
    const sm::float3 T1 = (lensq > 0.0F) ? (sm::float3(-Vh.y, Vh.x, 0.0F) / std::sqrt(lensq))
                                        : sm::float3(1.0F, 0.0F, 0.0F);
    const sm::float3 T2 = sm::cross(Vh, T1);
    const float r = std::sqrt(u1);
    const float phi = Math::k2Pi * u2;
    const float t1 = r * std::cos(phi);
    float t2 = r * std::sin(phi);
    const float s = 0.5F * (1.0F + Vh.z);
    t2 = ((1.0F - s) * std::sqrt(std::max(0.0F, 1.0F - (t1 * t1)))) + (s * t2);
    const sm::float3 Nh = (t1 * T1) + (t2 * T2) + (std::sqrt(std::max(0.0F, 1.0F - (t1 * t1) - (t2 * t2))) * Vh);
    return sm::normalize(sm::float3(alpha * Nh.x, alpha * Nh.y, std::max(0.0F, Nh.z)));
}

// Single-scatter GGX microfacet reflection term (no MS compensation), scalar F0 — used to verify
// Helmholtz reciprocity of the underlying physical lobe.
[[nodiscard]] float singleScatterMicrofacet(float f0, const sm::float3& wo, const sm::float3& wi, float alpha) noexcept {
    if (wo.z <= 0.0F || wi.z <= 0.0F) {
        return 0.0F;
    }
    const sm::float3 h = sm::normalize(wo + wi);
    const float NoH = std::max(h.z, 0.0F);
    const float VoH = std::max(sm::dot(wo, h), 0.0F);
    if (NoH <= 0.0F || VoH <= 0.0F) {
        return 0.0F;
    }
    const float D = ggxD(NoH, alpha);
    const float G = smithG2(wi, wo, alpha);
    const float F = fresnelSchlick(f0, VoH);
    return (D * G * F) / std::max(4.0F * wi.z * wo.z, 1.0e-5F);
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
        const sm::float3 m = sampleUniformHemisphere(dist(rng), dist(rng));
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
        const sm::float3 l = sampleUniformHemisphere(dist(rng), dist(rng));
        const sm::float3 v = sampleUniformHemisphere(dist(rng), dist(rng));
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
        sm::float3(0.0F, 0.0F, 1.0F),
        sm::normalize(sm::float3(0.5F, 0.0F, 0.8660254F)),
        sm::normalize(sm::float3(0.8F, 0.0F, 0.6F)),
    };

    for (const float roughness : {0.0F, 0.25F, 0.5F, 0.9F}) {
        for (const sm::float3 wo : outgoingDirections) {
            double reflectance = 0.0;
            for (int i = 0; i < sampleCount; ++i) {
                const sm::float3 wi = sampleUniformHemisphere(dist(rng), dist(rng));
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
    mat.subsurfaceColorWeight = sm::float4(0.0F, 0.0F, 0.0F, 0.8F);
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
    const sm::float3 wo = sm::normalize(sm::float3(0.25F, 0.15F, 0.955F));

    std::array<GpuMaterial, 5> configs{};

    configs[0] = makeMaterial();
    configs[0].baseColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[0].baseMetalnessDiffRough = sm::float4(0.0F, 0.35F, 0.0F, 0.0F);
    configs[0].specularRoughAnisoIor = sm::float4(0.25F, 0.0F, 1.5F, 0.0F);

    configs[1] = configs[0];
    configs[1].baseColorWeight.w = 0.0F;
    configs[1].coatColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[1].coatRoughAnisoIorDark = sm::float4(0.15F, 0.0F, 1.4F, 0.25F);

    configs[2] = configs[0];
    configs[2].baseColorWeight.w = 0.0F;
    configs[2].fuzzColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[2].fuzzRoughPad.x = 0.4F;

    configs[3] = makeMaterial();
    configs[3].baseColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[3].transmissionColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    configs[3].transmissionParams = sm::float4(0.0F, 0.0F, 0.0F, 0.0F);
    configs[3].specularRoughAnisoIor = sm::float4(0.02F, 0.0F, 1.52F, 0.0F);

    configs[4] = makeMaterial();
    configs[4].baseColorWeight = sm::float4(0.0F, 0.0F, 0.0F, 0.0F);
    configs[4].subsurfaceColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
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
            const float a = mx_zeltner_sheen_dir_albedo(c, r);
            EXPECT_GE(a, 0.0F);
            EXPECT_LE(a, 1.0F);
        }
    }
    const float grazing = mx_zeltner_sheen_dir_albedo(0.05F, 0.4F);
    const float normalInc = mx_zeltner_sheen_dir_albedo(0.98F, 0.4F);
    EXPECT_GT(grazing, normalInc);

    // The sheen lobe itself must be finite, non-negative, and brighter at a grazing exit than
    // at a near-normal exit for a fixed grazing view (the velvet rim-light signature).
    const sm::float3 wo = sm::normalize(sm::float3(0.9F, 0.0F, 0.2F));
    const sm::float3 wiGrazing = sm::normalize(sm::float3(-0.9F, 0.0F, 0.15F));
    const sm::float3 wiNormal = sm::normalize(sm::float3(0.0F, 0.0F, 1.0F));
    const sm::float3 white(1.0F);
    const sm::float3 sGrazing = evalSheen(white, 0.4F, wo, wiGrazing);
    const sm::float3 sNormal = evalSheen(white, 0.4F, wo, wiNormal);
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
    const float smooth = mx_ggx_dir_albedo(kNoV, 0.02F);
    const float mid = mx_ggx_dir_albedo(kNoV, 0.25F);
    const float rough = mx_ggx_dir_albedo(kNoV, 0.9F);

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
    const sm::float3 wo = sm::normalize(sm::float3(0.25F, 0.15F, 0.955F));

    for (const float roughness : {0.2F, 0.4F, 0.6F, 0.85F}) {
        GpuMaterial metal = makeMaterial();
        metal.baseColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
        metal.baseMetalnessDiffRough = sm::float4(1.0F, 0.0F, 0.0F, 0.0F); // metalness = 1
        metal.specularColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);    // F82 tint = 1 (mirror)
        metal.specularRoughAnisoIor = sm::float4(roughness, 0.0F, 1.5F, 0.0F);

        const double energy = estimateWhiteFurnaceEnergy(metal, wo, 40000);
        EXPECT_GE(energy, 0.93) << "roughness=" << roughness << " energy=" << energy;
        EXPECT_LE(energy, 1.07) << "roughness=" << roughness << " energy=" << energy;
    }
}

TEST(Bsdf, GgxMultiScatterCompensationUsesFssNotF0) {
    // MaterialX-correct dielectric compensation uses Fss = F0 + (F90-F0)/21 (F90=1), not bare F0.
    // For a dielectric F0=0.04 that is Fss=0.0857 (~2x). The compensation stays small (the layer
    // is thin) but is materially larger than the legacy bare-F0 form, and bounded.
    const sm::float3 dielectricF0(0.04F);
    const sm::float3 Fss = dielectricF0 + (sm::float3(1.0F) - dielectricF0) * (1.0F / 21.0F);
    EXPECT_NEAR(Fss.x, 0.04F + 0.96F / 21.0F, 1.0e-5F);
    for (const float roughness : {0.2F, 0.5F, 0.9F}) {
        const float alpha = std::max(roughness * roughness, 0.001F);
        const sm::float3 comp = ggxMultiScatterCompensation(dielectricF0, 0.7F, alpha);
        const float Ess = mx_ggx_dir_albedo(0.7F, alpha);
        const float expected = 1.0F + Fss.x * (1.0F - Ess) / std::max(Ess, 1.0e-3F);
        EXPECT_NEAR(comp.x, expected, 1.0e-5F) << "roughness=" << roughness;
        EXPECT_GE(comp.x, 1.0F) << "roughness=" << roughness;
        EXPECT_LT(comp.x, 1.10F) << "roughness=" << roughness; // bounded for a thin dielectric layer
        // And it must exceed the legacy bare-F0 form (proving Fss, not F0, is wired in).
        const float legacy = 1.0F + dielectricF0.x * (1.0F - Ess) / std::max(Ess, 1.0e-3F);
        EXPECT_GT(comp.x, legacy) << "roughness=" << roughness;
    }
}

TEST(Bsdf, FresnelGgxDirAlbedoHonorsAlphaAndMatchesMaterialx) {
    // The Fresnel-weighted GGX directional albedo must (a) equal the MaterialX construction
    // F0*A + F90*B (F90=1) from the SAME rational fit as mx_ggx_dir_albedo, (b) honor alpha
    // (differ from the alpha-independent Schlick form `f0+(1-f0)*(1-NdotV)^5` at non-trivial
    // roughness), and (c) approach Schlick as alpha->0 (smooth dielectric ≈ Fresnel reflectance).
    // For a dielectric (f0=0.04) the albedo DECREASES with roughness: the white-furnace `A` term
    // drops with roughness (masking-shadowing) faster than the `B` (F90) term compensates, so a
    // rough dielectric layer reflects less → more energy reaches the diffuse substrate.
    const float f0 = 0.04F;
    constexpr float kNoV = 0.7F;
    const float smooth = fresnelGgxDirAlbedo(kNoV, f0, 0.04F);
    const float rough = fresnelGgxDirAlbedo(kNoV, f0, 0.81F);

    // (a) exact MaterialX construction F0*A + 1*B from the shared fit:
    const sm::float2 abSmooth = mx_ggx_dir_albedo_ab(kNoV, 0.04F);
    const sm::float2 abRough = mx_ggx_dir_albedo_ab(kNoV, 0.81F);
    EXPECT_NEAR(smooth, f0 * abSmooth.x + abSmooth.y, 1.0e-5F);
    EXPECT_NEAR(rough, f0 * abRough.x + abRough.y, 1.0e-5F);

    // (b) honors alpha: at high roughness it diverges from the alpha-independent Schlick form.
    const float schlick = f0 + (1.0F - f0) * std::pow(1.0F - kNoV, 5.0F);
    EXPECT_NEAR(fresnelGgxDirAlbedo(kNoV, f0, 0.02F), schlick, 2.0e-3F); // (c) smooth ≈ Schlick
    EXPECT_GT(std::abs(rough - schlick), 0.01F) << "rough form must differ from Schlick (alpha honored)";

    // (d) range + monotonic direction (decreasing with roughness for a dielectric).
    EXPECT_GE(smooth, 0.0F); EXPECT_LE(smooth, 1.0F);
    EXPECT_GE(rough, 0.0F); EXPECT_LE(rough, 1.0F);
    EXPECT_LT(rough, smooth) << "dielectric Fresnel-weighted albedo decreases with roughness";
}

TEST(Bsdf, TransmissionEvalPdfWeightMatchesPbrt) {
    // A3: with tint=1 and depth=0, the transmission throughput weight f*|NoL|/pdf must equal the
    // PBRT-v4 DielectricBxDF Radiance-mode weight (1-F)*(G/G1)/etap² times the MS 1/Ess boost.
    // This is the exact eval/pdf consistency invariant the old reflection-denom form violated
    // (it was off by ~(VoH+eta*LoH)²/(4*VoH), ~1.56x at normal incidence, eta 1.5). wi is built by
    // refracting wo through the macro normal N=(0,0,1) — a valid in-support transmitted direction.
    const sm::float3 one(1.0F);
    for (const float eta : {1.2F, 1.5F, 2.0F}) {
        for (const float rough : {0.08F, 0.3F, 0.7F}) {
            const float alpha = std::max(rough * rough, 0.001F);
            for (const float cosO : {0.25F, 0.6F, 0.95F}) {
                const sm::float3 wo = sm::normalize(sm::float3(0.35F, 0.2F, cosO));
                const float sin2O = std::max(0.0F, 1.0F - wo.z * wo.z);
                const float sin2T = sin2O / (eta * eta);
                if (sin2T >= 1.0F) continue; // macro TIR — skip
                const float cosT = std::sqrt(1.0F - sin2T);
                const sm::float3 wi = sm::normalize(sm::float3(-wo.x / eta, -wo.y / eta, -cosT));
                if (wo.z * wi.z >= 0.0F) continue; // sanity: opposite hemispheres

                const sm::float3 f = evalTransmissionMicrofacet(one, 0.0F, eta, alpha, alpha, wo, wi);
                const float pdf = transmissionPdf(eta, alpha, alpha, wo, wi);
                ASSERT_GT(pdf, 0.0F) << "eta=" << eta << " rough=" << rough << " cosO=" << cosO;
                ASSERT_GT(f.x, 0.0F);

                // Independently recompute wm, F, G, G1, etap, msComp.
                const float etap = wo.z > 0.0F ? eta : (1.0F / eta);
                sm::float3 wm = sm::normalize(wi * etap + wo);
                if (wm.z < 0.0F) wm = -wm;
                const float F = fresnelDielectric(std::abs(sm::dot(wo, wm)), eta);
                const float G = smithG2(wi, wo, alpha);
                const float G1wo = smithG1(std::max(wo.z, 0.0F), alpha);
                const float Ess = mx_ggx_dir_albedo(std::abs(wo.z), alpha);
                const float msComp = 1.0F / std::max(Ess, 1.0e-3F);
                const float expected = (1.0F - F) * (G / std::max(G1wo, 1.0e-6F)) / (etap * etap) * msComp;
                const float weight = f.x * std::abs(wi.z) / pdf;
                EXPECT_NEAR(weight, expected, 5.0e-3F * expected + 1.0e-4F)
                    << "eta=" << eta << " rough=" << rough << " cosO=" << cosO;
            }
        }
    }
}

TEST(Bsdf, ThinFilmGuardReturnsBaseWhenInactive) {
    // The eval path applies iridescence only when thin_film_weight>0 AND thickness>0;
    // otherwise the base Schlick reflectance is used unchanged. Verify that contract
    // (mirrors evalReflectionMicrofacetThinFilm / the thinFilmTint guard).
    auto applyThinFilm = [](sm::float3 baseF0, float cosT, float thickness, float ior, float weight) {
        const sm::float3 base = baseF0 + (sm::float3(1.0F) - baseF0) * std::pow(std::clamp(1.0F - cosT, 0.0F, 1.0F), 5.0F);
        if (weight <= 0.0F || thickness <= 0.0F) {
            return base;
        }
        const sm::float3 irid = thinFilmIridescentReflectance(baseF0, cosT, thickness, ior);
        return sm::lerp(base, irid, std::clamp(weight, 0.0F, 1.0F));
    };
    const sm::float3 baseF0(0.95F, 0.78F, 0.40F);
    for (const float cosT : {0.2F, 0.5F, 0.9F}) {
        const sm::float3 base = baseF0 + (sm::float3(1.0F) - baseF0) * std::pow(1.0F - cosT, 5.0F);
        const sm::float3 offThickness = applyThinFilm(baseF0, cosT, 0.0F, 1.5F, 1.0F);
        const sm::float3 offWeight = applyThinFilm(baseF0, cosT, 500.0F, 1.5F, 0.0F);
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
                const sm::float3 r = thinFilmIridescentReflectance(sm::float3(0.04F), cosT, thickness, filmIor);
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
    auto chroma = [](sm::float3 c) {
        const float s = c.x + c.y + c.z + 1.0e-6F;
        return sm::float2(c.x / s, c.y / s);
    };
    const sm::float3 darkBase(0.04F); // dielectric base maximises interference contrast
    const sm::float2 a = chroma(thinFilmIridescentReflectance(darkBase, 0.7F, 300.0F, 1.4F));
    const sm::float2 b = chroma(thinFilmIridescentReflectance(darkBase, 0.7F, 550.0F, 1.4F));
    const sm::float2 c = chroma(thinFilmIridescentReflectance(darkBase, 0.7F, 800.0F, 1.4F));
    EXPECT_GT(sm::length(a - b), 0.02F) << "300nm vs 550nm should differ in hue";
    EXPECT_GT(sm::length(b - c), 0.02F) << "550nm vs 800nm should differ in hue";
}

TEST(Bsdf, ThinFilmConductorIsVividAndFinite) {
    // The OpenPBR conductor thin-film (complex IOR via Gulbrandsen) must produce a STRONGER,
    // hue-shifting iridescence on a metal base than the dielectric Schlick approximation —
    // this is the anodized-metal vividness the real-IOR path could not reach.
    auto chroma = [](sm::float3 col) {
        const float s = col.x + col.y + col.z + 1.0e-6F;
        return sm::float2(col.x / s, col.y / s);
    };
    auto saturation = [](sm::float3 col) {
        const float mx = std::max({col.x, col.y, col.z});
        const float mn = std::min({col.x, col.y, col.z});
        return (mx > 1.0e-5F) ? (mx - mn) / mx : 0.0F;
    };
    const sm::float3 baseColor(0.55F, 0.56F, 0.58F); // neutral chromium-like metal
    const sm::float3 F82(1.0F);
    sm::float3 n, k;
    mx_artistic_ior(baseColor, F82, n, k);
    ASSERT_TRUE(std::isfinite(n.x) && std::isfinite(k.x));
    EXPECT_GT(sm::length(k), 0.0F) << "a reflective metal must have non-zero extinction";

    float maxSat = 0.0F;
    sm::float2 prevChroma(0.0F);
    float maxHueShift = 0.0F;
    bool first = true;
    for (const float thickness : {100.0F, 240.0F, 380.0F, 520.0F, 660.0F}) {
        const sm::float3 cond = mx_fresnel_airy(0.7F, true, baseColor, F82, n, k, thickness, 2.0F);
        ASSERT_TRUE(std::isfinite(cond.x) && std::isfinite(cond.y) && std::isfinite(cond.z))
            << "thickness=" << thickness;
        EXPECT_GE(std::min({cond.x, cond.y, cond.z}), 0.0F);
        EXPECT_LE(std::max({cond.x, cond.y, cond.z}), 1.2F);
        maxSat = std::max(maxSat, saturation(cond));
        const sm::float2 ch = chroma(cond);
        if (!first) {
            maxHueShift = std::max(maxHueShift, sm::length(ch - prevChroma));
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

// ─────────────────────────────────────────────────────────────────────────────────────────
// V0 — OpenPBR numeric-conformance suite.
//
// Hyperion's BSDF is the assumed ground truth that Theia is aligned to; OpenPBR's canonical
// reference implementation is MaterialX (`mx_*` genGLSL). These tests make "OpenPBR-compliant"
// measurable: they validate the analytic MaterialX fits we ported against ground-truth
// Monte-Carlo integration of the actual BRDFs, and assert physical-correctness properties
// (Helmholtz reciprocity, LTC energy normalization, hemispherical energy conservation) that
// the white-furnace-only checks cannot prove on their own.
// ─────────────────────────────────────────────────────────────────────────────────────────

TEST(Bsdf, OpenPbrV0_GgxDirAlbedoFitMatchesVndfIntegration) {
    // `mx_ggx_dir_albedo` is a verbatim port of MaterialX `mx_ggx_dir_albedo_analytic` (F0=F90=1).
    // Validate that fit against the directional albedo obtained by Monte-Carlo integrating the
    // single-scatter GGX BRDF (F=1) with VNDF sampling, whose estimator is mean(G2/G1). This is
    // exactly the energy the H2 multiple-scattering compensation recovers, so the fit must track
    // the true integral across the (NdotV, roughness) domain.
    std::mt19937 rng(0xC0FFEEU);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    constexpr int kSamples = 200000;
    for (const float nDotV : {0.2F, 0.5F, 0.85F}) {
        const sm::float3 V(std::sqrt(std::max(0.0F, 1.0F - (nDotV * nDotV))), 0.0F, nDotV);
        for (const float alpha : {0.3F, 0.6F, 1.0F}) {
            const float g1v = smithG1(V.z, alpha);
            double sum = 0.0;
            for (int i = 0; i < kSamples; ++i) {
                const sm::float3 H = sampleGgxVndf(V, alpha, dist(rng), dist(rng));
                const sm::float3 L = sm::normalize((2.0F * sm::dot(V, H) * H) - V);
                if (L.z <= 0.0F || g1v <= 0.0F) {
                    continue; // shadowed sample contributes zero to the albedo
                }
                sum += static_cast<double>(smithG2(L, V, alpha) / g1v);
            }
            const double integrated = sum / static_cast<double>(kSamples);
            const double fit = static_cast<double>(mx_ggx_dir_albedo(nDotV, alpha));
            EXPECT_NEAR(fit, integrated, 0.03) << "NdotV=" << nDotV << " alpha=" << alpha;
        }
    }
}

TEST(Bsdf, OpenPbrV0_GeneralizedSchlickF82IsF0AtNormalAndRisesToGrazing) {
    // Anchor the OpenPBR generalized-Schlick F82 conductor Fresnel (mx_fresnel_F82, mirroring the
    // shader's mx_fresnel_F82). Physical anchors that must hold:
    //  * at normal incidence reflectance == F0 (a dielectric with F0=0.04 reflects ~4%, NOT ~96%);
    //  * it rises monotonically toward ~1 at grazing;
    //  * with an F82 tint < 1 it dips below plain Schlick near the 82-degree peak.
    // (Regression guard for the fixed oracle Fresnel, which previously used mix(F0,F82,1-F) and
    // returned ~0.96 at normal for F82=1, turning full-specular dielectrics into mirrors.)
    const sm::float3 f0(0.04F);
    const sm::float3 whiteTint(1.0F);
    const sm::float3 atNormal = mx_fresnel_F82(f0, whiteTint, 1.0F);
    EXPECT_NEAR(atNormal.x, 0.04F, 2.0e-3F);
    const sm::float3 atGrazing = mx_fresnel_F82(f0, whiteTint, 0.02F);
    EXPECT_GT(atGrazing.x, 0.8F);
    // Monotonic rise from normal to grazing.
    float prev = -1.0F;
    for (const float mu : {1.0F, 0.8F, 0.6F, 0.4F, 0.2F, 0.05F}) {
        const float v = mx_fresnel_F82(f0, whiteTint, mu).x;
        EXPECT_GE(v, prev - 1.0e-3F) << "mu=" << mu;
        prev = v;
    }
    // F82 tint < 1 must reduce reflectance near the 82-degree peak (mu = cos(82deg) ~ 0.1392)
    // relative to a pure-white (F82=1) edge.
    const float muPeak = std::cos(82.0F * Math::kPi / 180.0F);
    const sm::float3 tinted = mx_fresnel_F82(f0, sm::float3(0.5F), muPeak);
    const sm::float3 untinted = mx_fresnel_F82(f0, whiteTint, muPeak);
    EXPECT_LT(tinted.x, untinted.x);
}

TEST(Bsdf, OpenPbrV0_ZeltnerSheenLtcIsEnergyNormalized) {
    // The Zeltner LTC sheen lobe (`mx_zeltner_sheen_brdf`, a faithful port of MaterialX
    // `mx_zeltner_sheen_brdf`) is a linear-cosine transform that must integrate to 1 over the
    // hemisphere. That normalization is precisely what makes evalSheen's directional albedo
    // equal `mx_zeltner_sheen_dir_albedo` — the value used for the view-dependent base-layer attenuation. If
    // the LTC coefficient fits (aInv/bInv) were wrong, this integral would drift from 1.
    std::mt19937 rng(0x5EED01U);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    constexpr int kSamples = 400000;
    for (const float nDotV : {0.15F, 0.5F, 0.9F}) {
        const sm::float3 V(std::sqrt(std::max(0.0F, 1.0F - (nDotV * nDotV))), 0.0F, nDotV);
        for (const float rough : {0.2F, 0.5F, 0.9F}) {
            double sum = 0.0;
            for (int i = 0; i < kSamples; ++i) {
                const sm::float3 L = sampleUniformHemisphere(dist(rng), dist(rng));
                // mx_zeltner_sheen_brdf already includes the cosine; uniform-hemisphere pdf = 1/2pi.
                sum += static_cast<double>(mx_zeltner_sheen_brdf(V, L, rough)) * (2.0 * Math::kPi);
            }
            const double integral = sum / static_cast<double>(kSamples);
            EXPECT_NEAR(integral, 1.0, 0.04) << "NdotV=" << nDotV << " rough=" << rough;
        }
    }
}

TEST(Bsdf, OpenPbrV0_EonDiffuseIsReciprocal) {
    // Helmholtz reciprocity: the OpenPBR base diffuse (EON/Fujii) BRDF must satisfy
    // f(wo,wi) == f(wi,wo) for all directions and roughnesses.
    const sm::float3 color(0.82F, 0.51F, 0.33F);
    const std::array<sm::float3, 4> dirs = {
        sm::normalize(sm::float3(0.30F, 0.10F, 0.95F)),
        sm::normalize(sm::float3(-0.55F, 0.40F, 0.73F)),
        sm::normalize(sm::float3(0.12F, -0.62F, 0.78F)),
        sm::normalize(sm::float3(0.70F, 0.20F, 0.68F)),
    };
    for (const float rough : {0.0F, 0.4F, 1.0F}) {
        for (const sm::float3& wo : dirs) {
            for (const sm::float3& wi : dirs) {
                const sm::float3 a = evalDiffuse(color, rough, wo, wi);
                const sm::float3 b = evalDiffuse(color, rough, wi, wo);
                EXPECT_NEAR(a.x, b.x, 1.0e-4F) << "rough=" << rough;
                EXPECT_NEAR(a.y, b.y, 1.0e-4F) << "rough=" << rough;
                EXPECT_NEAR(a.z, b.z, 1.0e-4F) << "rough=" << rough;
            }
        }
    }
}

TEST(Bsdf, OpenPbrV0_SingleScatterMicrofacetIsReciprocal) {
    // Helmholtz reciprocity of the single-scatter GGX specular lobe: D and the NoL*NoV
    // denominator are symmetric, Smith G2 is symmetric, and V·H == L·H for the half-vector, so
    // f(wo,wi) == f(wi,wo). (The Kulla-Conty/Turquin MS compensation is intentionally applied
    // per-view and is excluded here; that single-sided form is the MaterialX/Filament approach.)
    const std::array<sm::float3, 4> dirs = {
        sm::normalize(sm::float3(0.25F, 0.15F, 0.96F)),
        sm::normalize(sm::float3(-0.50F, 0.35F, 0.79F)),
        sm::normalize(sm::float3(0.40F, -0.45F, 0.80F)),
        sm::normalize(sm::float3(0.62F, 0.10F, 0.78F)),
    };
    for (const float f0 : {0.04F, 0.5F, 0.95F}) {
        for (const float alpha : {0.05F, 0.3F, 0.8F}) {
            for (const sm::float3& wo : dirs) {
                for (const sm::float3& wi : dirs) {
                    const float a = singleScatterMicrofacet(f0, wo, wi, alpha);
                    const float b = singleScatterMicrofacet(f0, wi, wo, alpha);
                    EXPECT_NEAR(a, b, std::max(1.0e-4F, 1.0e-3F * std::abs(a)))
                        << "f0=" << f0 << " alpha=" << alpha;
                }
            }
        }
    }
}

TEST(Bsdf, OpenPbrV0_SingleLobeMaterialsConserveEnergy) {
    // OpenPBR energy conservation for single-dominant-lobe materials — the regimes the current
    // additive lobe composition handles correctly: a pure conductor (metalness=1, with H2
    // multiple-scattering compensation), a pure dielectric diffuse base (specular off), and a
    // pure fuzz/sheen surface. White-furnace reflectance must stay within [0, ~1]. NOTE: the
    // per-view Turquin/Kulla-Conty MS compensation can overshoot ~15% at grazing on a smooth
    // high-albedo conductor (it conserves on the hemispherical average, not per-view), so the
    // bound is 1.20 rather than 1.0.
    const std::array<sm::float3, 3> views = {
        sm::normalize(sm::float3(0.0F, 0.0F, 1.0F)),
        sm::normalize(sm::float3(0.45F, 0.0F, 0.89F)),
        sm::normalize(sm::float3(0.70F, 0.20F, 0.68F)),
    };
    std::vector<GpuMaterial> mats;
    for (const float rough : {0.1F, 0.5F, 0.9F}) {
        GpuMaterial metal = makeMaterial();          // pure conductor
        metal.baseColorWeight = sm::float4(0.9F, 0.85F, 0.8F, 1.0F);
        metal.baseMetalnessDiffRough = sm::float4(1.0F, 0.4F, 0.0F, 0.0F);
        metal.specularColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
        metal.specularRoughAnisoIor = sm::float4(rough, 0.0F, 1.5F, 0.0F);
        mats.push_back(metal);

        GpuMaterial diff = makeMaterial();           // pure diffuse dielectric (specular off)
        diff.baseColorWeight = sm::float4(0.9F, 0.85F, 0.8F, 1.0F);
        diff.baseMetalnessDiffRough = sm::float4(0.0F, rough, 0.0F, 0.0F);
        diff.specularColorWeight = sm::float4(0.0F, 0.0F, 0.0F, 0.0F);
        mats.push_back(diff);
    }
    GpuMaterial fuzz = makeMaterial();               // pure fuzz/sheen
    fuzz.baseColorWeight = sm::float4(0.0F, 0.0F, 0.0F, 0.0F);
    fuzz.specularColorWeight = sm::float4(0.0F);
    fuzz.fuzzColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    fuzz.fuzzRoughPad = sm::float4(0.4F, 0.0F, 0.0F, 0.0F);
    mats.push_back(fuzz);

    for (size_t i = 0; i < mats.size(); ++i) {
        for (const sm::float3& wo : views) {
            const double energy = estimateWhiteFurnaceEnergy(mats[i], wo, 20000);
            ASSERT_TRUE(std::isfinite(energy)) << "material " << i;
            EXPECT_GE(energy, 0.0) << "material " << i;
            EXPECT_LE(energy, 1.20) << "material " << i;
        }
    }
}

TEST(Bsdf, OpenPbrV0_LayeredMaterialsConserveEnergy) {
    // OpenPBR energy conservation for LAYERED materials: a full-weight dielectric specular over
    // the diffuse base, and a clear-coat over a dielectric or metal base. With Step 1's
    // directional-albedo layer coupling (substrate attenuated by 1 - E_above, symmetric in
    // NoV/NoL), the stack conserves: white-furnace reflectance stays within [0, 1.20] (per-view
    // conductor MS-compensation headroom only). Tighter than the additive ~1.25 of v0.5.0.
    const std::array<sm::float3, 3> views = {
        sm::normalize(sm::float3(0.0F, 0.0F, 1.0F)),
        sm::normalize(sm::float3(0.45F, 0.0F, 0.89F)),
        sm::normalize(sm::float3(0.70F, 0.20F, 0.68F)),
    };
    for (const float metalness : {0.0F, 1.0F}) {
        for (const float rough : {0.1F, 0.5F, 0.9F}) {
            for (const bool coated : {false, true}) {
                GpuMaterial mat = makeMaterial();
                mat.baseColorWeight = sm::float4(0.9F, 0.85F, 0.8F, 1.0F);
                mat.baseMetalnessDiffRough = sm::float4(metalness, 0.4F, 0.0F, 0.0F);
                mat.specularColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F); // full dielectric specular
                mat.specularRoughAnisoIor = sm::float4(rough, 0.0F, 1.5F, 0.0F);
                if (coated) {
                    mat.coatColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
                    mat.coatRoughAnisoIorDark = sm::float4(0.1F, 0.0F, 1.5F, 0.25F);
                }
                for (const sm::float3& wo : views) {
                    const double energy = estimateWhiteFurnaceEnergy(mat, wo, 20000);
                    ASSERT_TRUE(std::isfinite(energy)) << "metalness=" << metalness << " rough="
                                                       << rough << " coated=" << coated;
                    EXPECT_GE(energy, 0.0);
                    EXPECT_LE(energy, 1.20) << "metalness=" << metalness << " rough=" << rough
                                            << " coated=" << coated;
                }
            }
        }
    }
}

// ── V0 composition-level conformance ─────────────────────────────────────────────────────
// Per-lobe tests above can all pass while *combined* materials are wrong (a layer eats too much
// or too little of the layer below, or the additive mix double-counts). These tests exercise the
// full evalBSDF on multi-lobe materials and assert the cross-lobe invariants: full-BSDF Helmholtz
// reciprocity, bounded white-furnace energy per composition, and that stacking a coat never
// *creates* energy versus the same base uncoated. This is the V0 gate that closes Steps 1–4: when
// proper directional-albedo layering (Step 1) lands, the per-view bound should tighten toward 1.0.

[[nodiscard]] sm::float3 evalCompositeReflection(const GpuMaterial& mat, const sm::float3& wo,
                                                const sm::float3& wi) noexcept {
    const sm::float3 N(0.0F, 0.0F, 1.0F);
    const sm::float3 T(1.0F, 0.0F, 0.0F);
    const sm::float3 B(0.0F, 1.0F, 0.0F);
    return evalBSDF(mat, wo, wi, N, T, B, N, T, B);
}

TEST(Bsdf, OpenPbrV0_FullBsdfIsReciprocalForOpaqueCompositions) {
    // Helmholtz reciprocity must hold for the COMBINED material, not just each lobe: coat over
    // dielectric, coat over conductor, and specular+diffuse coupling. A layering scheme that
    // attenuates the base by a view-dependent factor must apply it symmetrically or reciprocity
    // breaks. The coat/sheen base-layer scale is itself view-dependent, so this is a real check.
    // (The single-scatter lobes are exactly reciprocal; the per-view Kulla-Conty MS compensation
    // is deliberately single-sided, so the conductor base carries ~1% asymmetry — tolerated here.)
    const std::array<sm::float3, 4> dirs = {
        sm::normalize(sm::float3(0.30F, 0.10F, 0.95F)),
        sm::normalize(sm::float3(-0.55F, 0.40F, 0.73F)),
        sm::normalize(sm::float3(0.40F, -0.45F, 0.80F)),
        sm::normalize(sm::float3(0.62F, 0.10F, 0.78F)),
    };
    std::vector<GpuMaterial> mats;

    GpuMaterial coatDielectric = makeMaterial();   // clear coat over diffuse+specular dielectric
    coatDielectric.baseColorWeight = sm::float4(0.8F, 0.6F, 0.4F, 1.0F);
    coatDielectric.baseMetalnessDiffRough = sm::float4(0.0F, 0.4F, 0.0F, 0.0F);
    coatDielectric.specularRoughAnisoIor = sm::float4(0.3F, 0.0F, 1.5F, 0.0F);
    coatDielectric.coatColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    coatDielectric.coatRoughAnisoIorDark = sm::float4(0.15F, 0.0F, 1.5F, 0.25F);
    mats.push_back(coatDielectric);

    GpuMaterial coatConductor = makeMaterial();    // clear coat over conductor
    coatConductor.baseColorWeight = sm::float4(0.9F, 0.85F, 0.6F, 1.0F);
    coatConductor.baseMetalnessDiffRough = sm::float4(1.0F, 0.4F, 0.0F, 0.0F);
    coatConductor.specularRoughAnisoIor = sm::float4(0.25F, 0.0F, 1.5F, 0.0F);
    coatConductor.coatColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
    coatConductor.coatRoughAnisoIorDark = sm::float4(0.1F, 0.0F, 1.5F, 0.25F);
    mats.push_back(coatConductor);

    GpuMaterial specDiffuse = makeMaterial();      // specular + diffuse coupling
    specDiffuse.baseColorWeight = sm::float4(0.7F, 0.5F, 0.55F, 1.0F);
    specDiffuse.baseMetalnessDiffRough = sm::float4(0.0F, 0.5F, 0.0F, 0.0F);
    specDiffuse.specularRoughAnisoIor = sm::float4(0.35F, 0.0F, 1.5F, 0.0F);
    mats.push_back(specDiffuse);

    for (size_t i = 0; i < mats.size(); ++i) {
        for (const sm::float3& wo : dirs) {
            for (const sm::float3& wi : dirs) {
                const sm::float3 a = evalCompositeReflection(mats[i], wo, wi);
                const sm::float3 b = evalCompositeReflection(mats[i], wi, wo);
                const float tol = std::max(3.0e-4F, 2.0e-2F * std::max(a.x, b.x));
                EXPECT_NEAR(a.x, b.x, tol) << "material " << i;
                EXPECT_NEAR(a.y, b.y, tol) << "material " << i;
                EXPECT_NEAR(a.z, b.z, tol) << "material " << i;
            }
        }
    }
}

TEST(Bsdf, OpenPbrV0_CoatNeverCreatesEnergyOverUncoatedBase) {
    // Stacking a clear coat redistributes energy (some reflected at the coat, the rest attenuated
    // into the base); it must NOT increase total reflectance over the same base uncoated by more
    // than the coat's own Fresnel reflectance. This catches additive double-counting in layering.
    const std::array<sm::float3, 3> views = {
        sm::normalize(sm::float3(0.0F, 0.0F, 1.0F)),
        sm::normalize(sm::float3(0.45F, 0.0F, 0.89F)),
        sm::normalize(sm::float3(0.70F, 0.20F, 0.68F)),
    };
    for (const float metalness : {0.0F, 1.0F}) {
        for (const float rough : {0.1F, 0.5F, 0.9F}) {
            GpuMaterial base = makeMaterial();
            base.baseColorWeight = sm::float4(0.85F, 0.8F, 0.75F, 1.0F);
            base.baseMetalnessDiffRough = sm::float4(metalness, 0.4F, 0.0F, 0.0F);
            base.specularColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
            base.specularRoughAnisoIor = sm::float4(rough, 0.0F, 1.5F, 0.0F);
            GpuMaterial coated = base;
            coated.coatColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
            coated.coatRoughAnisoIorDark = sm::float4(0.1F, 0.0F, 1.5F, 0.25F);
            for (const sm::float3& wo : views) {
                const double eBase = estimateWhiteFurnaceEnergy(base, wo, 20000);
                const double eCoat = estimateWhiteFurnaceEnergy(coated, wo, 20000);
                // The coat replaces base energy (symmetric transmittance) instead of adding it,
                // so coated must not exceed uncoated by more than the coat's own reflectance.
                EXPECT_LE(eCoat, eBase + 0.15) << "metalness=" << metalness << " rough=" << rough;
            }
        }
    }
}

TEST(Bsdf, OpenPbrV0_RoughTransmissionFurnaceStaysBounded) {
    // Rough dielectric transmission has no MS compensation yet (Step 2). It must still never
    // CREATE energy: full-furnace reflectance+transmittance stays bounded as roughness rises
    // (energy is lost, not gained). Smooth glass sits near 1; rough glass loses to the bound.
    const std::array<sm::float3, 2> views = {
        sm::normalize(sm::float3(0.0F, 0.0F, 1.0F)),
        sm::normalize(sm::float3(0.55F, 0.0F, 0.84F)),
    };
    for (const float rough : {0.02F, 0.3F, 0.7F}) {
        GpuMaterial glass = makeMaterial();
        glass.baseColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
        glass.transmissionColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
        glass.transmissionParams = sm::float4(0.0F);
        glass.specularRoughAnisoIor = sm::float4(rough, 0.0F, 1.5F, 0.0F);
        for (const sm::float3& wo : views) {
            const double energy = estimateWhiteFurnaceEnergy(glass, wo, 24000);
            ASSERT_TRUE(std::isfinite(energy)) << "rough=" << rough;
            EXPECT_GE(energy, 0.0) << "rough=" << rough;
            EXPECT_LE(energy, 1.20) << "rough=" << rough;
        }
    }
}

TEST(Bsdf, OpenPbrV0_RoughTransmissionMsCompensationBoostsLobe) {
    // Step 2: the rough dielectric BTDF gets a 1/Ess multiple-scattering boost. Verify the factor
    // is > 1 and grows with roughness (more lost single-scatter energy to recover), and that the
    // boosted transmitted lobe is finite/non-negative and brighter than the uncompensated lobe.
    float prev = 1.0F;
    for (const float alpha : {0.05F, 0.3F, 0.8F}) {
        const float ess = mx_ggx_dir_albedo(0.7F, alpha);
        const float comp = 1.0F / std::max(ess, 1.0e-3F);
        EXPECT_GE(comp, 1.0F) << "alpha=" << alpha;
        EXPECT_GE(comp, prev - 1.0e-3F) << "alpha=" << alpha;
        prev = comp;
        const sm::float3 wo = sm::normalize(sm::float3(0.0F, 0.0F, 1.0F));
        const sm::float3 wi = sm::normalize(sm::float3(0.2F, 0.0F, -0.95F));
        const sm::float3 t = evalTransmissionMicrofacet(sm::float3(1.0F), 0.0F, 1.5F, alpha, alpha, wo, wi);
        EXPECT_TRUE(std::isfinite(t.x)) << "alpha=" << alpha;
        EXPECT_GE(t.x, 0.0F) << "alpha=" << alpha;
    }
}

TEST(Bsdf, OpenPbrV0_ThinFilmUnderCoatStaysBounded) {
    // Thin-film iridescence beneath a clear coat is a layered composition (coat Fresnel over airy
    // base). The combined material must stay finite, non-negative, and energy-bounded across a
    // thickness sweep — the coat must not amplify the iridescent base into energy creation.
    const sm::float3 wo = sm::normalize(sm::float3(0.3F, 0.1F, 0.95F));
    for (const float thicknessNm : {100.0F, 300.0F, 500.0F, 700.0F}) {
        GpuMaterial mat = makeMaterial();
        mat.baseColorWeight = sm::float4(0.9F, 0.85F, 0.6F, 1.0F);
        mat.baseMetalnessDiffRough = sm::float4(1.0F, 0.3F, 0.0F, 0.0F);
        mat.specularRoughAnisoIor = sm::float4(0.2F, 0.0F, 1.5F, 0.0F);
        mat.thinFilmParams = sm::float4(1.0F, thicknessNm, 2.0F, 0.0F);
        mat.coatColorWeight = sm::float4(1.0F, 1.0F, 1.0F, 1.0F);
        mat.coatRoughAnisoIorDark = sm::float4(0.1F, 0.0F, 1.5F, 0.25F);
        const double energy = estimateWhiteFurnaceEnergy(mat, wo, 20000);
        ASSERT_TRUE(std::isfinite(energy)) << "thickness=" << thicknessNm;
        EXPECT_GE(energy, 0.0) << "thickness=" << thicknessNm;
        EXPECT_LE(energy, 1.25) << "thickness=" << thicknessNm;
    }
}

// ── Step 3: bulk subsurface random-walk primitives (mirror of Harmonia bsdf_shared.slang) ──

[[nodiscard]] float sssExtinction(float radius, const sm::float3& radiusScale) noexcept {
    const float mfp = std::max(radius, 1.0e-4F) *
                      std::max(Math::luminance(sm::clamp(radiusScale, sm::float3(0.0F), sm::float3(1.0F))), 1.0e-4F);
    return 1.0F / std::max(mfp, 1.0e-4F);
}

[[nodiscard]] sm::float3 sampleHenyeyGreenstein(const sm::float3& wi, float g, sm::float2 xi) noexcept {
    float cosTheta;
    if (std::abs(g) < 1.0e-3F) {
        cosTheta = 1.0F - 2.0F * xi.x;
    } else {
        const float sqrTerm = (1.0F - g * g) / (1.0F + g - 2.0F * g * xi.x);
        cosTheta = -(1.0F + g * g - sqrTerm * sqrTerm) / (2.0F * g);
    }
    const float sinTheta = std::sqrt(std::max(0.0F, 1.0F - cosTheta * cosTheta));
    const float phi = 2.0F * Math::kPi * xi.y;
    sm::float3 a = std::abs(wi.x) > 0.9F ? sm::float3(0.0F, 1.0F, 0.0F) : sm::float3(1.0F, 0.0F, 0.0F);
    const sm::float3 t = sm::normalize(sm::cross(a, wi));
    const sm::float3 b = sm::cross(wi, t);
    // Z-component negated to match the forward-continuation mean-cosine convention (see
    // bsdf_shared.slang's mirror of this function for the derivation).
    const sm::float3 localDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), -cosTheta);
    return sm::normalize(t * localDir.x + b * localDir.y + wi * localDir.z);
}

TEST(Bsdf, OpenPbrV0_SssExtinctionIsPositiveAndDecreasesWithRadius) {
    // Larger mean-free-path (radius × radius_scale) -> smaller extinction (less dense medium).
    const float e1 = sssExtinction(0.1F, sm::float3(1.0F));
    const float e2 = sssExtinction(1.0F, sm::float3(1.0F));
    const float e3 = sssExtinction(10.0F, sm::float3(1.0F));
    EXPECT_GT(e1, e2);
    EXPECT_GT(e2, e3);
    EXPECT_GT(e3, 0.0F);
    // A brighter (higher-luminance) radius_scale widens the mean free path -> lower extinction.
    const float eDim = sssExtinction(1.0F, sm::float3(0.2F));
    const float eBright = sssExtinction(1.0F, sm::float3(1.0F));
    EXPECT_GT(eDim, eBright);
}

TEST(Bsdf, OpenPbrV0_HenyeyGreensteinMeanCosineMatchesAnisotropyG) {
    // The Henyey-Greenstein phase function's defining property: the mean cosine of the
    // scattered direction (relative to the incoming travel direction continuing forward)
    // equals the anisotropy parameter g. This is exactly what OpenPBR's
    // subsurface_scatter_anisotropy is specified to control, so it anchors the mapping.
    std::mt19937 rng(0xA55A5EEDU);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const sm::float3 wi = sm::normalize(sm::float3(0.3F, -0.6F, 0.74F));
    constexpr int kSamples = 200000;
    for (const float g : {-0.7F, -0.2F, 0.0F, 0.3F, 0.8F}) {
        double sum = 0.0;
        for (int i = 0; i < kSamples; ++i) {
            const sm::float3 wo = sampleHenyeyGreenstein(wi, g, sm::float2(dist(rng), dist(rng)));
            ASSERT_TRUE(std::isfinite(wo.x) && std::isfinite(wo.y) && std::isfinite(wo.z)) << "g=" << g;
            sum += static_cast<double>(sm::dot(wo, wi));
        }
        const double meanCos = sum / static_cast<double>(kSamples);
        EXPECT_NEAR(meanCos, static_cast<double>(g), 0.01) << "g=" << g;
    }
}

// Faithful mirror of the SHADER's signed dielectric Fresnel (Harmonia math.slang
// fresnelDielectric): unlike the air→medium-only oracle helper above, this branches on the
// sign of cosI so it also models the medium→air side and reports total internal reflection.
// This is the exact function closesthit.slang's shadeMediumBoundary relies on, so the test
// below locks the medium-exit / TIR behaviour that a naive abs()-based version silently broke.
[[nodiscard]] float fresnelDielectricSigned(float cosI, float eta) noexcept {
    cosI = std::clamp(cosI, -1.0F, 1.0F);
    float ei = 1.0F;
    float et = eta;
    if (cosI <= 0.0F) {
        cosI = std::abs(cosI);
        ei = eta;
        et = 1.0F;
    }
    const float ratio = ei / et;
    const float sinT2 = ratio * ratio * std::max(0.0F, 1.0F - cosI * cosI);
    if (sinT2 >= 1.0F) {
        return 1.0F; // total internal reflection
    }
    const float cosT = std::sqrt(std::max(0.0F, 1.0F - sinT2));
    const float rs = (ei * cosI - et * cosT) / (ei * cosI + et * cosT);
    const float rp = (et * cosI - ei * cosT) / (et * cosI + ei * cosT);
    return 0.5F * (rs * rs + rp * rp);
}

TEST(Bsdf, OpenPbrV0_MediumExitFresnelReportsTirAboveCriticalAngle) {
    // Locks the Step 3/4 medium-boundary exit convention (closesthit.slang shadeMediumBoundary):
    // an inside→outside (medium→air) crossing must be evaluated with a NEGATED incidence cosine
    // so fresnelDielectric takes the medium→air branch and reports total internal reflection
    // beyond the critical angle. The original bug passed abs(cos), which always evaluated the
    // air→medium branch and NEVER reported TIR — letting light escape supercritically.
    for (const float eta : {1.33F, 1.5F, 1.8F}) {
        // Critical angle: sin(theta_c) = 1/eta  =>  cos(theta_c) = sqrt(1 - 1/eta^2).
        const float cosCrit = std::sqrt(std::max(0.0F, 1.0F - 1.0F / (eta * eta)));

        // At normal incidence (cos=1) the interface is well below critical: it must transmit
        // (F < 1) for BOTH the air-incidence and medium-incidence branches.
        EXPECT_LT(fresnelDielectricSigned(-1.0F, eta), 1.0F) << "eta=" << eta;
        EXPECT_LT(fresnelDielectricSigned(1.0F, eta), 1.0F) << "eta=" << eta;

        // Just STEEPER than critical (larger |cos|): medium→air still transmits (F < 1).
        const float cosBelow = std::min(1.0F, cosCrit + 0.05F);
        EXPECT_LT(fresnelDielectricSigned(-cosBelow, eta), 1.0F) << "eta=" << eta;

        // Just GRAZING past critical (smaller |cos|): medium→air must be TIR (F == 1) — the
        // exact case the abs() bug missed. The buggy air→medium branch (positive cosine) must
        // NOT report TIR at the same angle, which is why the sign matters.
        const float cosAbove = std::max(0.0F, cosCrit - 0.05F);
        EXPECT_FLOAT_EQ(fresnelDielectricSigned(-cosAbove, eta), 1.0F) << "eta=" << eta;
        EXPECT_LT(fresnelDielectricSigned(cosAbove, eta), 1.0F)
            << "air->medium must never TIR (documents why the exit cosine must be negated); eta=" << eta;

        // Reflect probability F and transmit probability (1-F) partition unity by construction —
        // the stochastic branch selection in shadeMediumBoundary relies on this.
        const float F = fresnelDielectricSigned(-0.9F, eta);
        EXPECT_NEAR(F + (1.0F - F), 1.0F, 1.0e-6F) << "eta=" << eta;
    }
}


