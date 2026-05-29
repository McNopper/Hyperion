import common;
import env;

[shader("miss")]
void backgroundMiss(inout RayPayload payload)
{
    // Write G-buffer sky sentinel for primary rays.
    if (payload.depth == 0u) {
        const uint2 pix = DispatchRaysIndex().xy;
        gNormal[pix] = float4(0.0, 0.0, 0.0, 0.0);
        gDepth[pix]  = kInf;
    }

    float3 Le = evalEnvironment(WorldRayDirection(), pushConst.envLuminanceScale);

    // MIS balance heuristic: weight BSDF path against competing env NEE from the previous surface.
    // When env importance sampling is active and this isn't a delta/specular bounce,
    // downweight the BSDF path contribution since env NEE already sampled the same light.
    // Ref: PBR Book 4th ed §13.4.3; Veach "Robust Monte Carlo Methods" §9.2
    float misW = 1.0;
    if (pushConst.envImportanceWidth > 0u && !payload.specularBounce && payload.depth > 0u) {
        const float envPdf  = evalEnvImportancePdf(WorldRayDirection());
        const float bsdfPdf = payload.lastBsdfPdf;
        if (envPdf > 0.0) {
            misW = bsdfPdf / (bsdfPdf + envPdf);
        }
    }

    payload.radiance += payload.throughput * Le * misW;
    payload.terminated = true;
}
