import common;
import math;

[shader("raygeneration")]
void pathTracingRaygen()
{
    uint2 launchID = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    uint seed = wangHash(launchID.x + launchID.y * launchDim.x + pushConst.rngSeed * launchDim.x * launchDim.y);
    float3 totalRadiance = 0.0.xxx;
    const uint spp = max(pushConst.samplesPerPixel, 1u);

    for (uint s = 0; s < spp; ++s) {
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

        for (uint depth = 0; depth < pushConst.maxDepth && !payload.terminated; ++depth) {
            RayDesc ray;
            ray.Origin = payload.origin;
            ray.Direction = payload.direction;
            ray.TMin = kRayEpsilon;
            ray.TMax = kInf;
            TraceRay(tlas, RAY_FLAG_NONE, kInstanceMaskAll, 0, 2, 0, ray, payload);
            payload.depth++;
        }

        // Firefly suppression: clamp per-sample luminance in display space (post-exposure).
        // Clamping after exposure multiplication makes the threshold camera-independent.
        float3 sampleRadiance = payload.radiance * camera.exposure;
        const float sampleLum = dot(sampleRadiance, kRec2020Luma);
        if (sampleLum > kFireflyClamp)
            sampleRadiance *= kFireflyClamp / sampleLum;

        totalRadiance += sampleRadiance;
        seed = payload.seed;
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
