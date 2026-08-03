// anyhit.slang — shadow-ray opacity resolution (OpenPBR `geometry_opacity`).
//
// Only the SHADOW hit groups carry an any-hit. Camera/indirect rays keep the
// stochastic pass-through gate in the shared estimator (that is the exact
// `mix(S_ambient-medium, M_surface, α)` mix), so their hit groups have no
// any-hit and traversal treats every candidate as opaque.
//
// Shadow rays instead accumulate the spec's "transparent shadows" transmittance
// (OpenPBR §Opacity / Transparency): a surface whose base is opaque transmits
// `1 - α`, so the ray multiplies (1-α) per crossing and stops at the first
// α = 1 hit. Without this a cut-out leaf casts the shadow of a solid leaf.
//
// Opacity micromaps ride on exactly this: microtriangles the bake proved
// uniformly transparent/opaque are resolved by traversal and never reach here;
// only the "unknown" (cut-out edge) microtriangles invoke the shader. The
// answer is identical either way — the micromap only removes work.

import common;
import surface;
import texsampler;

/// α at a triangle hit, from the interpolated UV. Position/normal/tangent are
/// deliberately not reconstructed: opacity needs the UV and nothing else.
float triangleOpacity(BuiltInTriangleIntersectionAttributes attr)
{
    GpuInstance instance = instances[(uint)InstanceID()];
    GpuMaterial mat = materials[instance.materialIndex];

    uint primitiveIndex = PrimitiveIndex();
    uint i0 = indices[instance.indexOffset + primitiveIndex * 3 + 0];
    uint i1 = indices[instance.indexOffset + primitiveIndex * 3 + 1];
    uint i2 = indices[instance.indexOffset + primitiveIndex * 3 + 2];

    float b1 = attr.barycentrics.x;
    float b2 = attr.barycentrics.y;
    float b0 = 1.0 - b1 - b2;
    float2 uv = vertices[i0].uv * b0 + vertices[i1].uv * b1 + vertices[i2].uv * b2;

    HyperionTextureSampler tex;
    return evalGeometryOpacity(mat, tex, uv);
}

/// α at an analytic-sphere hit. The UV parameterisation must match
/// `makeSphereHit` in closesthit.slang, or a textured mask would land
/// differently on the camera and shadow paths.
float sphereOpacity(SphereIntersectionAttributes attr)
{
    GpuInstance instance = instances[(uint)InstanceID()];
    GpuMaterial mat = materials[instance.materialIndex];

    float3x4 o2w = ObjectToWorld3x4();
    float3 center = mul(o2w, float4(0.0, 0.0, 0.0, 1.0));
    float3 position = WorldRayOrigin() + WorldRayDirection() * attr.t;
    float3 n = normalize(position - center);

    float phi = atan2(n.z, n.x);
    float theta = acos(clamp(n.y, -1.0, 1.0));
    float2 uv = float2((phi + kPi) / (2.0 * kPi), theta / kPi);

    HyperionTextureSampler tex;
    return evalGeometryOpacity(mat, tex, uv);
}

/// Fold one crossing into the shadow ray's transmittance. `AcceptHitAndEndSearch`
/// on a fully present surface makes `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`
/// behave exactly as before for opaque geometry (one accepted hit ends the
/// traversal, the shadow miss never runs, and the caller reads 0).
void accumulateShadowOpacity(inout ShadowPayload payload, float alpha)
{
    if (alpha >= 1.0) {
        payload.transmittance = 0.0;
        AcceptHitAndEndSearch();
    }
    payload.transmittance *= (1.0 - alpha);
    IgnoreHit();
}

[shader("anyhit")]
void triangleShadowAnyHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    accumulateShadowOpacity(payload, triangleOpacity(attr));
}

[shader("anyhit")]
void sphereShadowAnyHit(inout ShadowPayload payload, in SphereIntersectionAttributes attr)
{
    accumulateShadowOpacity(payload, sphereOpacity(attr));
}
