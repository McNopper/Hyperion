import common;
import math;

/// Sample the equirectangular IBL env map for the given direction (Rec.2020 linear).
float3 sampleEnvMap(float3 dir)
{
    float3 d = normalize(dir);
    float u = atan2(d.z, d.x) * (0.5 * kInvPi) + 0.5;
    float v = acos(clamp(d.y, -1.0, 1.0)) * kInvPi;  // v=0 at top (y=+1), v=1 at bottom (y=-1)
    return max(envMap.SampleLevel(envSampler, float2(u, v), 0).xyz, 0.0.xxx);
}

/// Evaluate environment radiance for a given direction (Rec.2020 linear, cd/m²).
/// Returns envLuminanceScale * IBL sample when an env map is loaded,
/// or black (0,0,0) for scenes without an env map.
float3 evalEnvironment(float3 direction, float envLuminanceScale)
{
    if (pushConst.hasEnvMap != 0u) {
        return envLuminanceScale * sampleEnvMap(direction);
    }
    return 0.0.xxx;
}

// ── Env map importance sampling via 2D separable CDF ──────────────────────────
// Ref: Pharr, Jakob & Humphreys — "Physically Based Rendering" 4th ed. §12.5
//      Veach — "Robust Monte Carlo Methods" §9.2 (MIS balance heuristic)

/// Binary search: find largest bin index i ∈ [0, n-1] such that cdf[base+i+1] <= r.
/// CDF has (n+1) entries at base[0..n] with cdf[base]=0, cdf[base+n]=1.
uint sampleCdf(StructuredBuffer<float> cdf, uint base, uint n, float r)
{
    uint lo = 0u, hi = n - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (cdf[base + mid + 1u] <= r) lo = mid + 1u;
        else                            hi = mid;
    }
    return lo;
}

/// Importance-sample the env map using the pre-built 2D separable CDF.
/// Uses continuous CDF inversion: r1/r2 determine both the selected bin and the
/// fractional offset within that bin, avoiding additional random numbers.
/// Returns the sampled world-space direction; pdfOmega is the solid-angle PDF.
///
/// Ref: PBR Book §12.5 — infinite area light sampling; §13.4.3 — MIS balance heuristic
float3 sampleEnvImportance(inout uint seed, out float pdfOmega)
{
    const uint W = pushConst.envImportanceWidth;
    const uint H = pushConst.envImportanceHeight;
    const float r1 = randFloat(seed);
    const float r2 = randFloat(seed);

    // Sample row via marginal CDF
    const uint row = sampleCdf(envMarginalCdf, 0u, H, r1);

    // Sample column via conditional CDF for the selected row
    const uint col = sampleCdf(envConditionalCdf, row * (W + 1u), W, r2);

    // Continuous inversion within selected bin (avoids extra random numbers)
    // Maps r1 → fractional v, r2 → fractional u within their respective cells
    const float rowLo = envMarginalCdf[row];
    const float rowHi = envMarginalCdf[row + 1u];
    const float tRow  = (rowHi > rowLo) ? clamp((r1 - rowLo) / (rowHi - rowLo), 0.0, 1.0) : 0.5;

    const uint  cBase = row * (W + 1u);
    const float colLo = envConditionalCdf[cBase + col];
    const float colHi = envConditionalCdf[cBase + col + 1u];
    const float tCol  = (colHi > colLo) ? clamp((r2 - colLo) / (colHi - colLo), 0.0, 1.0) : 0.5;

    const float uFrac = (float(col) + tCol) / float(W);
    const float vFrac = (float(row) + tRow) / float(H);

    // Discrete probability of this texel = pRow * pCol
    // Solid-angle PDF: p_omega = p_UV * W * H / (2 * pi^2 * sin(theta))
    // where p_UV = pRow * pCol is the probability density in [0,1]x[0,1] UV space
    const float pRow = rowHi - rowLo;
    const float pCol = colHi - colLo;
    const float theta = kPi * vFrac;
    const float sinTheta = max(sin(theta), 1.0e-5);
    pdfOmega = (pRow * pCol * float(W) * float(H)) / (2.0 * kPi * kPi * sinTheta);

    // Reconstruct direction from UV (matching sampleEnvMap convention):
    //   u = atan2(d.z, d.x) / (2*pi) + 0.5  →  phi = 2*pi*(u - 0.5)
    //   v = acos(d.y) / pi                   →  theta = pi * v
    const float phi = 2.0 * kPi * (uFrac - 0.5);
    float3 dir;
    dir.y = cos(theta);
    dir.x = sinTheta * cos(phi);
    dir.z = sinTheta * sin(phi);
    return normalize(dir);
}

/// Evaluate the solid-angle PDF for a given direction under the env importance distribution.
/// Used by the miss shader to compute the MIS weight for BSDF-sampled rays.
float evalEnvImportancePdf(float3 dir)
{
    const uint W = pushConst.envImportanceWidth;
    const uint H = pushConst.envImportanceHeight;

    const float3 d = normalize(dir);
    const float u = atan2(d.z, d.x) * (0.5 * kInvPi) + 0.5;
    const float v = acos(clamp(d.y, -1.0, 1.0)) * kInvPi;

    const uint col = min(uint(u * float(W)), W - 1u);
    const uint row = min(uint(v * float(H)), H - 1u);

    const float pRow = envMarginalCdf[row + 1u] - envMarginalCdf[row];
    const float pCol = envConditionalCdf[row * (W + 1u) + col + 1u]
                     - envConditionalCdf[row * (W + 1u) + col];

    const float theta    = kPi * (float(row) + 0.5) / float(H);
    const float sinTheta = max(sin(theta), 1.0e-5);
    return (pRow * pCol * float(W) * float(H)) / (2.0 * kPi * kPi * sinTheta);
}
