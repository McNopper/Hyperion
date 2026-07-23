import common;
import math;
import medium_walk;
import env;

[shader("miss")]
void backgroundMiss(inout RayPayload payload)
{
    // Step 3: no boundary was hit within the sampled free-flight distance inside a bulk
    // subsurface medium — a real scattering event occurs at the sampled point. This is NOT
    // an environment miss (the ray never left the object), so skip env lookup entirely.
    if (payload.insideMedium) {
        // Pure absorber: raygen set TMax = kInf (no free-flight sampled), so a miss means the ray
        // escaped the geometry without a boundary — unbounded absorber, transmittance exp(-σ_t·∞)=0.
        if (mediumIsPureAbsorber(payload.mediumAlbedo)) {
            payload.terminated = true;
            return;
        }
        mediumScatterEvent(payload.throughput, payload.seed, payload.terminated,
                           payload.mediumSigmaT, payload.mediumAlbedo, payload.mediumG,
                           WorldRayOrigin(), WorldRayDirection(), payload.mediumFreeDist,
                           payload.origin, payload.direction);
        return;
    }

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
