import common;
import math;

[shader("raygeneration")]
void pathTracingRaygen()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    uint pixelKey = launchID.x + launchID.y * launchDim.x;
    float3 totalRadiance = 0.0.xxx;
    const uint spp = max(pushConst.samplesPerPixel, 1u);

    for (uint s = 0; s < spp; ++s) {
        // A0(a): fresh Owen-Sobol' sample per (pixel, frame, spp-index); dim resets to 0 so
        // camera AA jitter takes the first low-discrepancy 2D, lens the next, path the rest.
        RngState seed = makeRngState(pixelKey, pushConst.rngSeed * spp + s);
        float2 jitter = float2(randFloat(seed), randFloat(seed));
        float2 uv = (float2(launchID) + jitter) / float2(launchDim);
        uv.y = 1.0 - uv.y;

        float4 clipPos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
        float4 viewPos = mul(camera.invProj, clipPos);
        viewPos /= max(viewPos.w, 1.0e-6);

        float3 rayDir = normalize(mul((float3x3)camera.invView, viewPos.xyz));
        float3 rayOrigin = camera.position.xyz;

        if (camera.lensRadius > 0.0) {
            float2 lens = camera.lensRadius * sampleDisk(randFloat2(seed));
            float3 lensWorld = mul((float3x3)camera.invView, float3(lens, 0.0));
            float3 focalPoint = rayOrigin + rayDir * camera.focusDistance;
            rayOrigin += lensWorld;
            rayDir = normalize(focalPoint - rayOrigin);
        }

        RayPayload payload;
        payload.radiance       = 0.0.xxx;
        payload.throughput     = 1.0.xxx;
        payload.origin         = rayOrigin;
        payload.direction      = rayDir;
        payload.depth          = 0u;
        payload.seed           = seed;
        payload.terminated     = false;
        payload.specularBounce = false;
        payload.lastBsdfPdf    = 0.0;
        payload.insideMedium   = false;
        payload.mediumSigmaT   = 0.0.xxx;
        payload.mediumAlbedo   = 0.0.xxx;
        payload.mediumG        = 0.0;
        payload.mediumEta      = 1.0;
        payload.mediumFreeDist = 0.0;

        // Surface-bounce budget vs volumetric-walk budget are SEPARATE. A dense subsurface/
        // transmission medium has a tiny mean free path, so a single surface bounce can expand
        // into hundreds of internal scatter/boundary segments; charging each against maxDepth
        // would starve the surrounding light transport (paths die inside the medium → the object
        // goes dark). Instead, medium segments consume their own large budget (kMaxMediumWalk)
        // and do NOT advance payload.depth, so only genuine SURFACE interactions count toward
        // maxDepth. A global iteration cap guarantees termination.
        const uint kMaxMediumWalk = 256u;
        uint surfaceBounces = 0u;
        uint totalIters = 0u;
        const uint maxIters = pushConst.maxDepth + kMaxMediumWalk;
        while (surfaceBounces < pushConst.maxDepth && !payload.terminated && totalIters < maxIters) {
            bool wasInMedium = payload.insideMedium;
            RayDesc ray;
            ray.Origin = payload.origin;
            ray.Direction = payload.direction;
            ray.TMin = wasInMedium ? kMediumEpsilon : kRayEpsilon;
            if (wasInMedium) {
                // G5/G6 hero-wavelength free-flight sampling: σ_t-importance-weighted channel
                // selection (A3(c) synergy: unbiased, matches Theia's gi.comp.slang). chProb is
                // proportional to extinction so near-transparent channels aren't over-sampled.
                // pdfColl/pdfSurf in miss.slang/closesthit.slang use the same chProb so the
                // balance-heuristic estimator is exactly unbiased. σ_t clamped to 1e-6 above.
                float3 st = max(payload.mediumSigmaT, 1.0e-6.xxx);
                float3 chProb = st / (st.x + st.y + st.z);
                float chSel = randFloat(payload.seed);
                float sigmaHero = chSel < chProb.x ? st.x : (chSel < chProb.x + chProb.y ? st.y : st.z);
                float u = randFloat(payload.seed);
                payload.mediumFreeDist = -log(max(1.0 - u, 1.0e-6)) / max(sigmaHero, 1.0e-6);
                ray.TMax = payload.mediumFreeDist;
            } else {
                ray.TMax = kInf;
            }
            TraceRay(tlas, RAY_FLAG_NONE, kInstanceMaskAll, 0, 2, 0, ray, payload);
            totalIters++;
            // Only an ordinary (non-medium) segment counts as a surface bounce toward maxDepth.
            if (!wasInMedium) {
                surfaceBounces++;
                payload.depth++;
            }
        }

        // NaN/Inf guard only: a single degenerate path must never contaminate the
        // accumulation. Hyperion is the unbiased reference — no firefly clamp.
        float3 sampleRadiance = max(payload.radiance * camera.exposure, 0.0.xxx);
        if (!all(isfinite(sampleRadiance)) || any(isnan(sampleRadiance))) {
            sampleRadiance = 0.0.xxx;
        }

        totalRadiance += sampleRadiance;
    }

    // Temporal accumulation: blend current sample into running average.
    // frameIndex==0 clears the buffer; subsequent frames blend progressively.
    // totalRadiance is already exposure-weighted (applied during firefly clamping).
    const float3 currentSample = totalRadiance / float(spp);
    if (pushConst.frameIndex == 0u) {
        hdrTarget[launchID] = float4(currentSample, 1.0);
    } else {
        const float3 prev = hdrTarget[launchID].rgb;
        const float t = 1.0 / float(pushConst.frameIndex + 1u);
        hdrTarget[launchID] = float4(lerp(prev, currentSample, t), 1.0);
    }
}
