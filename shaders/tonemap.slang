// Standalone bindings for the tone-mapping fragment pass.
// We do NOT import common.slang to avoid pulling in RaytracingAccelerationStructure
// (only valid in ray-tracing stages) and other ray-tracing-specific resources.

[[vk::binding(1, 0)]] RWTexture2D<float4> hdrTarget;

/// Matches GpuTypes.hpp::PushConstants (byte-for-byte layout).
struct PushConstants {
    uint  frameIndex;
    uint  maxDepth;
    uint  rngSeed;
    float envLuminanceScale;
    uint  lightCount;
    uint  outputColorSpace;  ///< OutputColorSpace enum (used by tonemap pass)
    uint  samplesPerPixel;
    uint  hasEnvMap;
};
[[vk::push_constant]] PushConstants pushConst;

// ---------------------------------------------------------------------------
// Color space conversion matrices (working space: linear Rec.2020)
// ---------------------------------------------------------------------------

/// Rec.2020 → Rec.709 (D65)
/// Coefficients from ITU-R BT.2087 / IEC 61966-2-1.
float3 rec2020ToRec709(float3 c) {
    return float3(
        dot(c, float3( 1.6605f, -0.5876f, -0.0728f)),
        dot(c, float3(-0.1246f,  1.1329f, -0.0083f)),
        dot(c, float3(-0.0182f, -0.1006f,  1.1187f))
    );
}

/// Rec.2020 → Display P3 D65
float3 rec2020ToP3(float3 c) {
    return float3(
        dot(c, float3( 1.2249f, -0.2247f,  0.0000f)),
        dot(c, float3(-0.0420f,  1.0419f,  0.0000f)),
        dot(c, float3( 0.0000f, -0.0197f,  1.0197f))
    );
}

// ---------------------------------------------------------------------------
// ACES AP1 (ACEScg) tone mapping
// ---------------------------------------------------------------------------
//
// Colour matrices derived from the ACES specification S-2014-003 published by
// the Academy of Motion Picture Arts and Sciences (AMPAS).  The RRT+ODT
// analytic fit is by Stephen Hill, published in the SIGGRAPH 2016 course
// "HDR Theory and Practice":
//   https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// The rational polynomial coefficients are a mathematical description of the
// tone curve; no copyrightable expression is reproduced.
//
// All matrices are row-major for Slang float3x3 (mul(M, v) = M·v).
// Row-major values are transposed from the GLM column-major form in
// src/hyperion/utils/ColorSpace.cpp, which holds the same numeric values.

/// Rec.2020 → ACEScg AP1
static const float3x3 kRec2020ToAP1 = float3x3(
    float3(0.6131324f,  0.3395255f,  0.0474491f),  // AP1.R
    float3(0.0701243f,  0.9163394f,  0.0135363f),  // AP1.G
    float3(0.0205076f,  0.1096098f,  0.8699926f)   // AP1.B
);

/// ACEScg AP1 → Rec.709 D65  (composed: AP1 → Rec.2020 → Rec.709)
static const float3x3 kAP1ToRec709 = float3x3(
    float3( 2.9090f, -1.6933f, -0.2159f),  // Rec.709.R
    float3(-0.3595f,  1.3712f, -0.0114f),  // Rec.709.G
    float3(-0.0447f, -0.2477f,  1.2925f)   // Rec.709.B
);

/// ACEScg AP1 → Display P3 D65  (composed: AP1 → Rec.2020 → P3)
static const float3x3 kAP1ToP3 = float3x3(
    float3( 2.1173f, -1.0180f, -0.0996f),  // P3.R
    float3(-0.2071f,  1.2150f, -0.0077f),  // P3.G
    float3(-0.0219f, -0.1540f,  1.1758f)   // P3.B
);

/// ACES RRT+ODT rational-polynomial approximation (Stephen Hill, 2016).
/// Input/output are ACEScg AP1 linear.
float3 acesRrtOdtFit(float3 v) {
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

/// Full ACES tone mapper: Rec.2020 linear → tone-mapped output in target primaries.
/// outputMat must be kAP1ToRec709 (for SDR / scRGB) or kAP1ToP3 (for Display P3).
float3 acesFitted(float3 c2020, float3x3 outputMat) {
    float3 ap1 = mul(kRec2020ToAP1, c2020);
    ap1 = max(ap1, 0.0f.xxx);        // clamp negatives before tone curve
    ap1 = acesRrtOdtFit(ap1);
    return max(mul(outputMat, ap1), 0.0f.xxx);
}

// ---------------------------------------------------------------------------
// Legacy / fallback tone mappers
// ---------------------------------------------------------------------------

/// Hable / Uncharted-2 filmic (John Hable, GDC 2010).
/// Kept for reference; ACES is preferred for new content.
float3 hablePartial(float3 x) {
    const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
float3 hableFilmic(float3 x) {
    const float3 white = hablePartial(float3(11.2f, 11.2f, 11.2f));
    return hablePartial(2.0f * x) / white;
}

/// Luminance-preserving Reinhard (Reinhard et al., "Photographic Tone
/// Reproduction for Digital Images", SIGGRAPH 2002).
/// Tone-maps luminance, preserves Rec.2020 chromaticity.
float3 reinhardLuminance(float3 x) {
    const float lum = dot(x, float3(0.2627f, 0.6780f, 0.0593f)); // Rec.2020 Kr/Kg/Kb
    if (lum <= 0.0f) return 0.0f.xxx;
    const float lumTm = lum / (1.0f + lum);
    return x * (lumTm / lum);
}

// ---------------------------------------------------------------------------
// Transfer functions (OETF)
// ---------------------------------------------------------------------------

/// SMPTE ST.2084 PQ OETF — input in absolute nits, output in [0, 1].
/// 1.0 = 10 000 nit peak (HDR10 maximum).
float pqOetf(float L) {
    const float m1 = 0.1593017578125f;
    const float m2 = 78.84375f;
    const float c1 = 0.8359375f;
    const float c2 = 18.8515625f;
    const float c3 = 18.6875f;
    float Yn  = saturate(L / 10000.0f);
    float Ym1 = pow(Yn, m1);
    return pow((c1 + c2 * Ym1) / (1.0f + c3 * Ym1), m2);
}
float3 pqOetf3(float3 c) { return float3(pqOetf(c.x), pqOetf(c.y), pqOetf(c.z)); }

/// ITU-R BT.2100 HLG OETF.
/// Input: scene-linear, 1.0 = HLG reference white (BT.2100 scene-referred).
/// Output in [0, 1]; 0.75 corresponds to reference white.
/// Constants: a = 0.17883277, b = 1 - 4a, c = 0.5 - a·ln(4a)  (BT.2100 Table 5).
float hlgOetf(float E) {
    const float a     = 0.17883277f;
    const float b     = 0.28466892f;
    const float c_hlg = 0.55991073f;
    E = max(0.0f, E);
    return (E <= (1.0f / 12.0f))
        ? sqrt(3.0f * E)
        : (a * log(12.0f * E - b) + c_hlg);
}
float3 hlgOetf3(float3 c) { return float3(hlgOetf(c.x), hlgOetf(c.y), hlgOetf(c.z)); }

/// sRGB / BT.709 OETF (IEC 61966-2-1).
float srgbOetf(float x) {
    x = max(0.0f, x);
    return (x <= 0.0031308f) ? (12.92f * x) : (1.055f * pow(x, 1.0f / 2.4f) - 0.055f);
}
float3 srgbOetf3(float3 c) { return float3(srgbOetf(c.x), srgbOetf(c.y), srgbOetf(c.z)); }

// Display P3 uses the same OETF curve as sRGB (IEC 61966-2-1).
float3 p3Oetf3(float3 c) { return srgbOetf3(c); }

// ---------------------------------------------------------------------------
// HDR10 highlight roll-off
// ---------------------------------------------------------------------------
//
// Scene-linear 1.0 maps to the ITU-R BT.2408-5 (2022) reference white of
// 203 nit.  A Reinhard-style luminance shoulder prevents hard clipping at the
// display peak (typically 1 000 nit for consumer HDR10 displays).

static const float kHDR10PaperWhiteNits = 203.0f;  // ITU-R BT.2408-5 reference white
static const float kHDR10KneeNits       = 600.0f;  // begin roll-off
static const float kHDR10PeakNits       = 1000.0f; // target display peak

/// Soft-clips per-channel luminance to kHDR10PeakNits via a Reinhard shoulder.
/// Preserves chromaticity by scaling uniformly by the luminance ratio.
float3 hdr10SoftClip(float3 nitsLinear) {
    const float lum = dot(nitsLinear, float3(0.2627f, 0.6780f, 0.0593f));
    if (lum <= 0.0f) return 0.0f.xxx;
    float lumOut;
    if (lum <= kHDR10KneeNits) {
        lumOut = lum;
    } else {
        // Reinhard shoulder mapped to [knee, peak]
        float t = (lum - kHDR10KneeNits) / (kHDR10PeakNits - kHDR10KneeNits);
        lumOut  = kHDR10KneeNits + (kHDR10PeakNits - kHDR10KneeNits) * t / (1.0f + t);
    }
    return nitsLinear * (lumOut / lum);
}

// ---------------------------------------------------------------------------
// Fragment entry point
// ---------------------------------------------------------------------------

[shader("fragment")]
float4 tonemapFrag(float4 svPos : SV_Position) : SV_Target0
{
    int2 coord = int2(svPos.xy);
    float3 hdr = max(hdrTarget[coord].xyz, 0.0f.xxx);
    float3 out3;

    // pushConst.outputColorSpace values (must match OutputColorSpace enum in GpuTypes.hpp):
    //   0 = HDR10     — PQ OETF (SMPTE ST.2084), BT.2020 primaries
    //   1 = HLG       — HLG OETF (ITU-R BT.2100), BT.2020 primaries
    //   2 = scRGB     — extended linear Rec.709; compositor applies display EOTF
    //   3 = Display P3 — ACES filmic, P3 D65 primaries, sRGB-like OETF
    //   4 = SDR linear — Rec.709, clamped, no OETF (debug/capture)
    //   5 = SDR       — ACES filmic, Rec.709 primaries, sRGB OETF  (default / fallback)
    switch (pushConst.outputColorSpace) {
        case 0: {
            // HDR10: scale scene-linear to nits, soft-clip highlight, apply PQ OETF.
            // Paper white 203 nit per ITU-R BT.2408-5; peak 1 000 nit for typical displays.
            float3 nits = hdr * kHDR10PaperWhiteNits;
            nits = hdr10SoftClip(nits);
            out3 = pqOetf3(nits);
            break;
        }
        case 1: {
            // HLG: scene-linear is already normalized to reference white = 1.0.
            // Apply HLG OETF directly (ITU-R BT.2100 scene-referred usage).
            out3 = saturate(hlgOetf3(hdr));
            break;
        }
        case 2: {
            // scRGB: Rec.2020 → Rec.709, output linear (extended values > 1 allowed).
            // The OS compositor / display driver applies the display EOTF.
            out3 = rec2020ToRec709(hdr);
            break;
        }
        case 3: {
            // Display P3: ACES filmic in Rec.2020 → P3 D65, then P3 OETF.
            float3 p3 = acesFitted(hdr, kAP1ToP3);
            out3 = p3Oetf3(saturate(p3));
            break;
        }
        case 4: {
            // SDR linear: Rec.2020 → Rec.709, clamped, no OETF.
            out3 = saturate(rec2020ToRec709(hdr));
            break;
        }
        default: {
            // SDR: ACES filmic in Rec.2020 → Rec.709, sRGB OETF.
            float3 sdr = acesFitted(hdr, kAP1ToRec709);
            out3 = srgbOetf3(saturate(sdr));
            break;
        }
    }

    return float4(out3, 1.0f);
}
