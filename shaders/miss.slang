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
        // Pure absorber (albedo == 0): no free-flight was sampled (raygen TMax = kInf), so a
        // miss means the ray left the geometry without striking a boundary (non-watertight
        // escape). The medium is then effectively unbounded along the ray and the exact
        // transmittance is exp(-σ_t·∞) = 0 — the path is fully absorbed.
        if (max(payload.mediumAlbedo.x, max(payload.mediumAlbedo.y, payload.mediumAlbedo.z)) == 0.0) {
            payload.terminated = true;
            return;
        }
        float3 scatterPos = WorldRayOrigin() + WorldRayDirection() * payload.mediumFreeDist;
        float3 newDir = sampleHenyeyGreenstein(WorldRayDirection(), payload.mediumG, randFloat2(payload.seed));

        // G5/G6 spectral scatter factor. A real scatter occurred at the sampled distance t. With
        // per-channel σ_t and hero-wavelength distance sampling, the throughput update is
        //   σ_s · T / pdf_coll,  T = exp(-σ_t·t),  σ_s = albedo·σ_t,
        //   pdf_coll = (1/3)·Σ_c σ_t_c·exp(-σ_t_c·t)   (balance heuristic over the 3 channels).
        // For achromatic σ_t this reduces EXACTLY to `throughput *= albedo` (the old scalar walk).
        float3 st = max(payload.mediumSigmaT, 1.0e-6.xxx);
        float3 T = exp(-st * payload.mediumFreeDist);
        float3 sigmaS = payload.mediumAlbedo * st;
        // A3(c) synergy: σ_t-importance collision pdf — matches chProb used in raygen.slang.
        // dot(chProb, st*T) = dot(st/(Σst), st*T) = Σ(st_c² * T_c) / Σst_c
        float3 chProb = st / (st.x + st.y + st.z);
        float pdfColl = dot(chProb, st * T);
        payload.throughput *= (sigmaS * T) / max(pdfColl, 1.0e-8);

        // Russian roulette on the walk's own throughput (depth-independent; the walk does not
        // consume the surface bounce budget — see raygen.slang), bounding cost for absorbing media.
        float q = saturate(max(payload.throughput.x, max(payload.throughput.y, payload.throughput.z)));
        if (randFloat(payload.seed) > q) {
            payload.terminated = true;
            return;
        }
        payload.throughput /= max(q, 1.0e-3);

        payload.origin = scatterPos + newDir * kMediumEpsilon;
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
