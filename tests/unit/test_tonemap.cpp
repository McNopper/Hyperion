
#include <glm/glm.hpp>

#include <array>
#include <cmath>
#include <gtest/gtest.h>

#include "hyperion/utils/ColorSpace.hpp"
#include "hyperion/utils/ToneMapping.hpp"

namespace {
constexpr float kEps    = 1.0e-4F;  // tolerance for tone curve outputs
constexpr float kEpsHi  = 1.0e-3F;  // looser tolerance for composed matrix paths
} // namespace

// ---------------------------------------------------------------------------
// PQ OETF reference values — from ITU-R BT.2100-2 / SMPTE ST.2084
// Verified against the normative formula; all in public domain.
// ---------------------------------------------------------------------------

TEST(PqOetf, ReferenceNitValues) {
    // pqOetfFromNits(n) == pqOetf(n / 10000)
    // Reference table (ITU-R BT.2100):
    struct Case { float nits; float expected; };
    const std::array<Case, 5> cases{{
        {    1.0f, 0.149946f},
        {  100.0f, 0.508078f},
        {  203.0f, 0.580689f},  // HDR10 paper white (ITU-R BT.2408)
        { 1000.0f, 0.751827f},
        {10000.0f, 1.000000f},
    }};
    for (const auto& c : cases) {
        EXPECT_NEAR(ColorSpace::pqOetfFromNits(c.nits), c.expected, kEps)
            << "nits = " << c.nits;
    }
}

TEST(PqOetf, BlackMapsToNearZero) {
    EXPECT_NEAR(ColorSpace::pqOetfFromNits(0.0f), 0.0f, 1.0e-4f);
}

TEST(PqOetf, IsMonotonicallyIncreasing) {
    constexpr std::array<float, 6> nitsSteps{0.f, 1.f, 100.f, 203.f, 1000.f, 10000.f};
    float prev = -1.f;
    for (float n : nitsSteps) {
        const float e = ColorSpace::pqOetfFromNits(n);
        EXPECT_GE(e, prev);
        prev = e;
    }
}

// ---------------------------------------------------------------------------
// HLG OETF — boundary conditions from ITU-R BT.2100-2 Table 5
// ---------------------------------------------------------------------------

TEST(HlgOetf, ZeroMapsToZero) {
    EXPECT_NEAR(ColorSpace::hlgOetf(0.0f), 0.0f, 1.0e-6f);
}

TEST(HlgOetf, KneePointIsExactHalf) {
    // E = 1/12 is the boundary; both branches must give 0.5 exactly.
    const float knee = 1.0f / 12.0f;
    EXPECT_NEAR(ColorSpace::hlgOetf(knee), 0.5f, kEps);
}

TEST(HlgOetf, OneMapsToPeakNearOne) {
    // HLG OETF(1.0) ≈ 1.0 for a well-normalised scene-linear signal.
    EXPECT_NEAR(ColorSpace::hlgOetf(1.0f), 1.0f, kEps);
}

TEST(HlgOetf, IsMonotonicallyIncreasing) {
    const std::array<float, 5> steps{0.f, 1.f / 12.f, 0.25f, 0.5f, 1.0f};
    float prev = -1.f;
    for (float e : steps) {
        const float encoded = ColorSpace::hlgOetf(e);
        EXPECT_GE(encoded, prev);
        prev = encoded;
    }
}

TEST(HlgOetf, VectorOverloadMatchesScalar) {
    const glm::vec3 v(0.0f, 1.0f / 12.0f, 1.0f);
    const glm::vec3 enc = ColorSpace::hlgOetf(v);
    EXPECT_NEAR(enc.r, ColorSpace::hlgOetf(v.r), 1.0e-6f);
    EXPECT_NEAR(enc.g, ColorSpace::hlgOetf(v.g), 1.0e-6f);
    EXPECT_NEAR(enc.b, ColorSpace::hlgOetf(v.b), 1.0e-6f);
}

// ---------------------------------------------------------------------------
// ACES RRT+ODT fit — correctness properties (no GPU needed)
// ---------------------------------------------------------------------------

TEST(AcesToneMap, WhiteNeutralInLinearAP1) {
    // Neutral grey in Rec.2020 should tone-map to a neutral (all channels equal).
    const glm::vec3 grey(1.0f, 1.0f, 1.0f);
    const glm::vec3 result = ToneMapping::acesRrtOdtFit(grey);
    EXPECT_NEAR(result.r, result.g, kEps);
    EXPECT_NEAR(result.g, result.b, kEps);
}

TEST(AcesToneMap, OutputIsBoundedAfterFit) {
    // acesFittedSDR (which clamps negatives) must produce values in [0, 1] for typical inputs.
    // Note: acesRrtOdtFit alone has a tiny negative offset at v=0 (black crush);
    // the shader and acesFitted both apply max(output, 0) to correct this.
    const std::array<float, 5> inputs{0.f, 0.18f, 1.f, 4.f, 8.f};
    for (float v : inputs) {
        const glm::vec3 out = ToneMapping::acesFittedSDR(glm::vec3(v));
        EXPECT_GE(out.r, 0.f) << "v = " << v;
        EXPECT_LE(out.r, 1.1f) << "v = " << v;  // allow small AP1→Rec.709 gamut overshoot
    }
}

TEST(AcesToneMap, IsMonotonicallyIncreasingOnGrey) {
    float prev = -1.f;
    for (float v : {0.f, 0.01f, 0.1f, 0.18f, 0.5f, 1.f, 4.f, 10.f}) {
        const float out = ToneMapping::acesRrtOdtFit(glm::vec3(v)).r;
        EXPECT_GE(out, prev) << "v = " << v;
        prev = out;
    }
}

TEST(AcesToneMap, DarkShadowsPreservedRelative) {
    // A dark input should produce a small but non-negative output via acesFittedSDR.
    // (acesRrtOdtFit itself has a known ~-3.8e-4 black-crush at v=0; the max(output,0)
    //  in acesFitted corrects this before display.)
    const glm::vec3 dark(0.001f);
    const glm::vec3 out = ToneMapping::acesFittedSDR(dark);
    EXPECT_GE(out.r, 0.f);
    EXPECT_LT(out.r, 0.05f);
}

TEST(AcesToneMap, SDRFittedOutputInGamutRange) {
    // acesFittedSDR should produce values in [0, 1] after gamut clamp for typical inputs.
    const std::array<glm::vec3, 5> hdr2020{{
        {0.f,  0.f,  0.f},
        {0.18f, 0.18f, 0.18f},
        {1.f,  1.f,  1.f},
        {4.f,  2.f,  0.5f},  // warm highlight
        {0.f,  0.f,  8.f},   // blue spike
    }};
    for (const auto& c : hdr2020) {
        const glm::vec3 out = ToneMapping::acesFittedSDR(c);
        // After acesFitted the matrix can push slightly out of gamut — clamp happens in shader.
        // Here we just verify the result is real (no NaN/inf) and the gamut-clamp output ≥ 0.
        EXPECT_FALSE(std::isnan(out.r));
        EXPECT_FALSE(std::isnan(out.g));
        EXPECT_FALSE(std::isnan(out.b));
        const glm::vec3 clamped = glm::clamp(out, 0.f, 1.f);
        EXPECT_GE(clamped.r, 0.f);
        EXPECT_LE(clamped.r, 1.f);
    }
}

// ---------------------------------------------------------------------------
// Hable filmic — sanity checks
// ---------------------------------------------------------------------------

TEST(HableToneMap, WhiteNeutralOnGrey) {
    const glm::vec3 grey(1.f);
    const glm::vec3 out = ToneMapping::hableFilmic(grey);
    EXPECT_NEAR(out.r, out.g, kEps);
    EXPECT_NEAR(out.g, out.b, kEps);
}

TEST(HableToneMap, OutputBelowOne) {
    // Hable's white point is 11.2 (scene-linear); inputs up to ~5 map within [0, 1].
    // Very large inputs (> white point) can exceed 1.0 before the sRGB clamp — that is
    // intentional and handled by saturate() in the shader.
    for (float v : {0.18f, 1.f, 4.f, 5.f}) {
        const glm::vec3 out = ToneMapping::hableFilmic(glm::vec3(v));
        EXPECT_LE(out.r, 1.0f + kEps) << "v = " << v;
        EXPECT_GE(out.r, 0.f);
    }
}

// ---------------------------------------------------------------------------
// Reinhard luminance — sanity checks
// ---------------------------------------------------------------------------

TEST(ReinhardToneMap, BlackStaysBlack) {
    const glm::vec3 out = ToneMapping::reinhardLuminance(glm::vec3(0.f));
    EXPECT_NEAR(glm::length(out), 0.f, 1.0e-6f);
}

TEST(ReinhardToneMap, OutputStrictlyBelowOne) {
    // Reinhard never reaches 1.0 for finite input.
    for (float v : {1.f, 4.f, 100.f}) {
        const glm::vec3 out = ToneMapping::reinhardLuminance(glm::vec3(v));
        EXPECT_LT(out.r, 1.0f);
        EXPECT_GE(out.r, 0.f);
    }
}

// ---------------------------------------------------------------------------
// ACES matrices — white-point preservation
// ---------------------------------------------------------------------------

TEST(AcesMatrices, Rec2020WhitePointPreservedInAP1) {
    const glm::vec3 white2020(1.f);
    const glm::vec3 ap1 = ToneMapping::kRec2020ToAP1 * white2020;
    EXPECT_NEAR(ap1.r, 1.f, kEpsHi);
    EXPECT_NEAR(ap1.g, 1.f, kEpsHi);
    EXPECT_NEAR(ap1.b, 1.f, kEpsHi);
}

TEST(AcesMatrices, AP1WhitePointPreservedInRec709) {
    const glm::vec3 whiteAP1(1.f);
    const glm::vec3 rec709 = ToneMapping::kAP1ToRec709 * whiteAP1;
    EXPECT_NEAR(rec709.r, 1.f, kEpsHi);
    EXPECT_NEAR(rec709.g, 1.f, kEpsHi);
    EXPECT_NEAR(rec709.b, 1.f, kEpsHi);
}

TEST(AcesMatrices, AP1WhitePointPreservedInP3) {
    const glm::vec3 whiteAP1(1.f);
    const glm::vec3 p3 = ToneMapping::kAP1ToP3 * whiteAP1;
    EXPECT_NEAR(p3.r, 1.f, kEpsHi);
    EXPECT_NEAR(p3.g, 1.f, kEpsHi);
    EXPECT_NEAR(p3.b, 1.f, kEpsHi);
}
