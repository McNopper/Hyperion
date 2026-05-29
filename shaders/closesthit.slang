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
        // Emissive mesh acts as an area light: add emission on every path bounce.
        // (No next-event estimation, so the emissive surface is sampled stochastically.)
        payload.radiance += payload.throughput * emission;
        payload.terminated = true;
        return;
    }

    uint seed = payload.seed;
    BSDFSample bsdfSample = sampleBSDF(mat, wo, N, T, B, seed);
    payload.seed = seed;
    if (bsdfSample.pdf <= 0.0 || max(bsdfSample.weight.x, max(bsdfSample.weight.y, bsdfSample.weight.z)) <= 0.0) {
        payload.terminated = true;
        return;
    }

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
