import common;
import math;
import bsdf;
import env;

HitInfo makeTriangleHit(BuiltInTriangleIntersectionAttributes attr)
{
    HitInfo hit;
    uint instanceIndex = (uint)InstanceID();
    GpuInstance instance = instances[instanceIndex];
    uint primitiveIndex = PrimitiveIndex();
    uint i0 = indices[instance.indexOffset + primitiveIndex * 3 + 0];
    uint i1 = indices[instance.indexOffset + primitiveIndex * 3 + 1];
    uint i2 = indices[instance.indexOffset + primitiveIndex * 3 + 2];

    GpuVertex v0 = vertices[i0];
    GpuVertex v1 = vertices[i1];
    GpuVertex v2 = vertices[i2];

    float b1 = attr.barycentrics.x;
    float b2 = attr.barycentrics.y;
    float b0 = 1.0 - b1 - b2;

    hit.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    hit.geoNormal = normalize(cross(v1.position - v0.position, v2.position - v0.position));
    hit.normal = normalize(v0.normal * b0 + v1.normal * b1 + v2.normal * b2);
    if (dot(hit.normal, hit.geoNormal) < 0.0) {
        hit.normal = -hit.normal;
    }
    hit.uv = v0.uv * b0 + v1.uv * b1 + v2.uv * b2;
    hit.instanceIndex = instanceIndex;
    hit.primitiveIndex = primitiveIndex;
    hit.frontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    return hit;
}

HitInfo makeSphereHit(SphereIntersectionAttributes attr)
{
    HitInfo hit;
    uint instanceIndex = (uint)InstanceID();
    GpuInstance instance = instances[instanceIndex];
    float3 center = vertices[instance.vertexOffset].position;

    hit.position = WorldRayOrigin() + WorldRayDirection() * attr.t;
    hit.geoNormal = normalize(hit.position - center);
    hit.normal = hit.geoNormal;

    float phi = atan2(hit.geoNormal.z, hit.geoNormal.x);
    float theta = acos(clamp(hit.geoNormal.y, -1.0, 1.0));
    hit.uv = float2((phi + kPi) / (2.0 * kPi), theta / kPi);
    hit.instanceIndex = instanceIndex;
    hit.primitiveIndex = 0u;
    hit.frontFace = dot(WorldRayDirection(), hit.geoNormal) < 0.0;
    return hit;
}

/// Fire a shadow ray toward emissive light `lightIdx` from surface point `origin`
/// (already offset by geometry normal epsilon). Returns true if the path is unoccluded.
bool traceNEEShadowRay(float3 origin, float3 direction, float tMax)
{
    ShadowPayload shadowPayload;
    shadowPayload.inShadow = true;
    RayDesc shadowRay;
    shadowRay.Origin    = origin;
    shadowRay.Direction = direction;
    shadowRay.TMin      = 0.0;
    shadowRay.TMax      = tMax;
    TraceRay(tlas,
             RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
             0xFF, 0, 2, 1, shadowRay, shadowPayload);
    return !shadowPayload.inShadow;
}

void shadeSurface(inout RayPayload payload, HitInfo hit)
{
    // Write G-buffer for primary rays before any early returns.
    if (payload.depth == 0u) {
        const uint2 pix = DispatchRaysIndex().xy;
        gNormal[pix] = float4(hit.normal, 0.0);
        gDepth[pix]  = RayTCurrent();
    }

    GpuInstance instance = instances[hit.instanceIndex];
    GpuMaterial mat = materials[instance.materialIndex];

    // Apply base color texture if bound (textureIndices.x != 0xFFFFFFFFu).
    if (mat.textureIndices.x != 0xFFFFFFFFu) {
        const uint texIdx = mat.textureIndices.x;
        mat.baseColorWeight.xyz =
            bindlessTextures[NonUniformResourceIndex(texIdx)]
                .SampleLevel(bindlessSamplers[NonUniformResourceIndex(texIdx)], hit.uv, 0.0).rgb;
    }

    float3 wo = -normalize(WorldRayDirection());
    float3 N = dot(wo, hit.normal) >= 0.0 ? hit.normal : -hit.normal;
    float3 geoN = dot(wo, hit.geoNormal) >= 0.0 ? hit.geoNormal : -hit.geoNormal;
    float3x3 tbn = buildTBN(N);
    float3 T = tbn[0];
    float3 B = tbn[1];

    float3 emission = max(mat.emissionColorLum.xyz, 0.0.xxx) * max(mat.emissionColorLum.w, 0.0);
    if (max(emission.x, max(emission.y, emission.z)) > 0.0 && dot(wo, hit.geoNormal) > 0.0) {
        // Add emissive contribution only for:
        //   • primary camera ray (depth == 0) — camera directly sees the light
        //   • after a specular/delta bounce — NEE cannot sample through perfect mirrors
        // All other direct hits are handled by NEE below and must not be double-counted.
        if (pushConst.emissiveLightCount == 0u || payload.depth == 0u || payload.specularBounce) {
            payload.radiance += payload.throughput * emission;
        }
        payload.terminated = true;
        return;
    }

    uint seed = payload.seed;

    // --- Next Event Estimation: bounding-sphere solid-angle sampling ---
    if (pushConst.emissiveLightCount > 0u) {
        uint lightIdx = min(uint(randFloat(seed) * float(pushConst.emissiveLightCount)),
                           pushConst.emissiveLightCount - 1u);
        GpuEmissiveLight eLight = emissiveLights[lightIdx];

        if (eLight.instanceIndex != hit.instanceIndex) {
            float3 toCenter = eLight.center - hit.position;
            float dist2     = dot(toCenter, toCenter);
            float dist      = sqrt(dist2);

            if (dist > eLight.radius + kRayEpsilon) {
                float sinThetaMax2 = (eLight.radius * eLight.radius) / dist2;
                float cosThetaMax  = sqrt(max(0.0, 1.0 - sinThetaMax2));
                float solidAngle   = 2.0 * kPi * (1.0 - cosThetaMax);

                // Sample a direction uniformly within the subtended cone.
                float xi0     = randFloat(seed);
                float xi1     = randFloat(seed);
                float cosT    = 1.0 - xi0 * (1.0 - cosThetaMax);
                float sinT    = sqrt(max(0.0, 1.0 - cosT * cosT));
                float phi     = 2.0 * kPi * xi1;

                float3x3 frame = buildFrameFromZ(toCenter / dist);
                float3 wi = normalize(frame[0] * (sinT * cos(phi)) +
                                      frame[1] * (sinT * sin(phi)) +
                                      frame[2] * cosT);

                float cosI = dot(wi, N);
                if (cosI > 0.0) {
                    // TMax stops just before the near side of the bounding sphere so the
                    // shadow ray never intersects the emissive mesh itself.
                    float tMax = max(kRayEpsilon, dist - eLight.radius - kRayEpsilon);
                    float3 shadowOrigin = hit.position + geoN * kRayEpsilon;
                    if (traceNEEShadowRay(shadowOrigin, wi, tMax)) {
                        float3 bsdfVal = evalBSDF(mat, wo, wi, N, T, B);
                        // Contribution = BSDF × Le × cos(θ) × solidAngle × N_lights
                        // (solidAngle = 1/pdf; N_lights cancels uniform light selection)
                        float3 nee = payload.throughput * bsdfVal * eLight.emission *
                                     cosI * solidAngle * float(pushConst.emissiveLightCount);
                        payload.radiance += nee;
                    }
                }
            }
        }
    }

    BSDFSample bsdfSample = sampleBSDF(mat, wo, N, T, B, seed);
    payload.seed = seed;
    if (bsdfSample.pdf <= 0.0 || max(bsdfSample.weight.x, max(bsdfSample.weight.y, bsdfSample.weight.z)) <= 0.0) {
        payload.terminated = true;
        return;
    }

    payload.specularBounce = bsdfSample.isDelta;
    payload.throughput *= bsdfSample.weight;
    payload.origin = hit.position + geoN * (dot(bsdfSample.direction, geoN) >= 0.0 ? kRayEpsilon : -kRayEpsilon);
    payload.direction = normalize(bsdfSample.direction);

    if (payload.depth >= 3u) {
        float q = saturate(max(payload.throughput.x, max(payload.throughput.y, payload.throughput.z)));
        if (randFloat(payload.seed) > q) {
            payload.terminated = true;
            return;
        }
        payload.throughput /= max(q, 1.0e-3);
    }
}

[shader("closesthit")]
void triangleClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    shadeSurface(payload, makeTriangleHit(attr));
}

[shader("closesthit")]
void sphereClosestHit(inout RayPayload payload, in SphereIntersectionAttributes attr)
{
    shadeSurface(payload, makeSphereHit(attr));
}
