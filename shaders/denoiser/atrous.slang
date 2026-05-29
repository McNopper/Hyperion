/// Edge-Avoiding À-Trous Wavelet Transform denoiser.
///
/// Algorithm: Dammertz et al. 2010 — "Edge-Avoiding À-Trous Wavelet Transform
/// for fast Global Illumination Filtering".  Public domain algorithm.
///
/// Each dispatch performs one wavelet level.  Call 5 times with stepWidth
/// 1, 2, 4, 8, 16 and ping-pong between two intermediate images.

[[vk::binding(0, 0)]] RWTexture2D<float4> inputColor;   ///< noisy HDR or previous level output
[[vk::binding(1, 0)]] RWTexture2D<float4> outputColor;  ///< filtered output for this level
[[vk::binding(2, 0)]] RWTexture2D<float4> gNormal;      ///< world-space normal (xyz); sky w=0
[[vk::binding(3, 0)]] RWTexture2D<float>  gDepth;       ///< ray-hit distance; sky = kInf

struct AtrousPushConstants {
    float sigmaColor;   ///< colour edge-stopping bandwidth (luminance)
    float sigmaNormal;  ///< normal edge-stopping exponent (power heuristic)
    float sigmaDepth;   ///< depth edge-stopping bandwidth (relative)
    int   stepWidth;    ///< current wavelet step: 1, 2, 4, 8, 16
    uint  imageWidth;
    uint  imageHeight;
};

[[vk::push_constant]] AtrousPushConstants pushConst;

/// 1-D separable B3-spline kernel  [1/16, 1/4, 3/8, 1/4, 1/16].
static const float kKernel[5] = { 1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };

static const float kInf = 1.0e30;
static const float kEps = 1.0e-6;

/// Rec.2020 luminance — coefficients from ITU-R BT.2020 (Kr=0.2627, Kg=0.6780, Kb=0.0593).
float luminance(float3 c) {
    return dot(c, float3(0.2627, 0.6780, 0.0593));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous(uint3 threadID : SV_DispatchThreadID)
{
    const int2 p = int2(threadID.xy);
    if (p.x >= int(pushConst.imageWidth) || p.y >= int(pushConst.imageHeight)) {
        return;
    }

    const float3 colorP  = inputColor[p].rgb;
    const float3 normalP = gNormal[p].xyz;
    const float  depthP  = gDepth[p];

    // Sky pixels: no geometry, pass through.
    if (depthP >= kInf * 0.5) {
        outputColor[p] = float4(colorP, 1.0);
        return;
    }

    float3 accumColor  = float3(0.0, 0.0, 0.0);
    float  accumWeight = 0.0;

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int2 q = clamp(
                p + int2(dx, dy) * pushConst.stepWidth,
                int2(0, 0),
                int2(int(pushConst.imageWidth) - 1, int(pushConst.imageHeight) - 1));

            const float3 colorQ  = inputColor[q].rgb;
            const float3 normalQ = gNormal[q].xyz;
            const float  depthQ  = gDepth[q];

            // B3-spline separable kernel weight.
            const float h = kKernel[dx + 2] * kKernel[dy + 2];

            // Colour (luminance) edge-stopping.
            const float lumDiff  = luminance(colorP) - luminance(colorQ);
            const float wColor   = exp(-(lumDiff * lumDiff) /
                                       (pushConst.sigmaColor * pushConst.sigmaColor + kEps));

            // Normal edge-stopping (power heuristic).
            const float normalDot = saturate(dot(normalP, normalQ));
            const float wNormal   = pow(normalDot, pushConst.sigmaNormal);

            // Depth edge-stopping (relative difference, skip sky).
            float wDepth = 1.0;
            if (depthQ < kInf * 0.5) {
                const float depthDiff = abs(depthP - depthQ) / (abs(depthP) + kEps);
                wDepth = exp(-(depthDiff * depthDiff) /
                              (pushConst.sigmaDepth * pushConst.sigmaDepth + kEps));
            }

            const float w = h * wColor * wNormal * wDepth;
            accumColor  += colorQ * w;
            accumWeight += w;
        }
    }

    outputColor[p] = float4(accumColor / max(accumWeight, kEps), 1.0);
}
