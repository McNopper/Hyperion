import common;
import math;
import bsdf_shared;
import env;

[shader("miss")]
void backgroundMiss(inout RayPayload payload)
{
    // Step 3: no boundary was hit within the sampled free-flight distance inside a bulk
    // subsurface medium — a real scattering event occurs at the sampled point. This is NOT
    // an environment miss (the ray never left the object), so skip env lookup entirely.
    if (payload.insideMedium) {
        float3 scatterPos = WorldRayOrigin() + WorldRayDirection() * payload.mediumFreeDist;
        float3 newDir = sampleHenyeyGreenstein(WorldRayDirection(), payload.mediumG, randFloat2(payload.seed));
        payload.throughput *= payload.mediumAlbedo;

        // Russian roulette on the walk's own throughput, matching the shared path
        // integrator's termination policy for ordinary surface bounces.
        if (payload.depth >= kRRMinDepth) {
            float q = saturate(max(payload.throughput.x, max(payload.throughput.y, payload.throughput.z)));
            if (randFloat(payload.seed) > q) {
                payload.terminated = true;
                return;
            }
            payload.throughput /= max(q, 1.0e-3);
        }

        payload.origin = scatterPos + newDir * kRayEpsilon;
        payload.direction = newDir;
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
