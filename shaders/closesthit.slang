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

    // UV-aligned tangent from triangle position/UV deltas (Lengyel's method).
    // Computed per-hit so normal mapping does not depend on baked vertex tangents.
    float2 duv1 = v1.uv - v0.uv;
    float2 duv2 = v2.uv - v0.uv;
    float3 dp1  = v1.position - v0.position;
    float3 dp2  = v2.position - v0.position;
    float  det  = duv1.x * duv2.y - duv2.x * duv1.y;
    hit.tangent = abs(det) > 1.0e-8 ? (dp1 * duv2.y - dp2 * duv1.y) / det : float3(1.0, 0.0, 0.0);

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
    // Tangent along increasing longitude (∂P/∂φ) for normal mapping.
    hit.tangent = normalize(float3(-hit.geoNormal.z, 0.0, hit.geoNormal.x));
    hit.instanceIndex = instanceIndex;
    hit.primitiveIndex = 0u;
    hit.frontFace = dot(WorldRayDirection(), hit.geoNormal) < 0.0;
    return hit;
}

/// Sentinel: material texture slot holds no texture (matches kNoTexture on the host).
static const uint kNoTexture = 0xFFFFFFFFu;

/// Sample a bindless texture (mip 0) by its descriptor index.
float4 sampleBindless(uint idx, float2 uv)
{
    return bindlessTextures[NonUniformResourceIndex(idx)]
        .SampleLevel(bindlessSamplers[NonUniformResourceIndex(idx)], uv, 0.0);
}

/// Fire a shadow ray from `origin` in `direction` up to `tMax`.
/// Returns true if the path is unoccluded (miss shader fired).
/// All instances are visible (kInstanceMaskAll); tMax is set to stop just
/// before the sampled light surface so the ray does not self-intersect it.
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
             kInstanceMaskAll, 0, 2, 1, shadowRay, shadowPayload);
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

    // ── Texture maps (bindless) — each slot is applied only when bound. ──────────
    // Slot layout: [0] base_color, [1] normal, [2] ORM, [3] emission.
    // Colour maps (base/emission) are pre-converted to linear Rec.2020 at load;
    // data maps (normal/ORM) are raw.

    // [0] base_color — replaces base_color tint.
    if (mat.textureIndices.x != kNoTexture) {
        mat.baseColorWeight.xyz = sampleBindless(mat.textureIndices.x, hit.uv).rgb;
    }
    // [2] ORM — glTF packing: G = roughness, B = metalness (R = occlusion, unused here).
    if (mat.textureIndices.z != kNoTexture) {
        float3 orm = sampleBindless(mat.textureIndices.z, hit.uv).rgb;
        mat.specularRoughAnisoIor.x = orm.g;   // specular_roughness
        mat.baseMetalnessDiffRough.x = orm.b;  // base_metalness
    }
    // [3] emission — modulates emission_color.
    if (mat.textureIndices.w != kNoTexture) {
        mat.emissionColorLum.xyz *= sampleBindless(mat.textureIndices.w, hit.uv).rgb;
    }

    float3 wo = -normalize(WorldRayDirection());
    float3 N = dot(wo, hit.normal) >= 0.0 ? hit.normal : -hit.normal;
    float3 geoN = dot(wo, hit.geoNormal) >= 0.0 ? hit.geoNormal : -hit.geoNormal;

    // [1] normal map — perturb the shading normal in a UV-aligned tangent frame.
    if (mat.textureIndices.y != kNoTexture) {
        float3 nTex = sampleBindless(mat.textureIndices.y, hit.uv).xyz * 2.0 - 1.0;
        float3 Tt = hit.tangent - N * dot(N, hit.tangent);
        if (dot(Tt, Tt) > 1.0e-12) {
            Tt = normalize(Tt);
            float3 Bt = cross(N, Tt);
            float3 Np = normalize(Tt * nTex.x + Bt * nTex.y + N * nTex.z);
            N = dot(wo, Np) >= 0.0 ? Np : N; // keep the perturbed normal on the visible side
        }
    }

    float3x3 tbn = buildTBN(N);
    float3 T = tbn[0];
    float3 B = tbn[1];

    // geometry_tangent [slot 5] — reorient the anisotropy frame. The map stores a
    // tangent direction (xy) in the UV-aligned tangent plane; rotate T/B accordingly.
    if (mat.textureIndices2.y != kNoTexture) {
        float2 tDir = sampleBindless(mat.textureIndices2.y, hit.uv).xy * 2.0 - 1.0;
        float3 Tuv = hit.tangent - N * dot(N, hit.tangent);
        if (dot(Tuv, Tuv) > 1.0e-12 && dot(tDir, tDir) > 1.0e-12) {
            Tuv = normalize(Tuv);
            float3 Buv = cross(N, Tuv);
            float3 Tn = normalize(Tuv * tDir.x + Buv * tDir.y);
            T = Tn - N * dot(N, Tn);
            if (dot(T, T) > 1.0e-12) {
                T = normalize(T);
                B = cross(N, T);
            } else {
                T = tbn[0];
                B = tbn[1];
            }
        }
    }

    // ── Coat frame (geometry_coat_normal [4], geometry_coat_tangent [6]) ──────────
    // Defaults to the base frame (incl. any geometry_tangent rotation) so unbound
    // slots reproduce the shared-frame result exactly.
    float3 coatN = N;
    float3 coatT = T;
    float3 coatB = B;
    if (mat.textureIndices2.x != kNoTexture) {
        float3 nTex = sampleBindless(mat.textureIndices2.x, hit.uv).xyz * 2.0 - 1.0;
        float3 Tt = hit.tangent - N * dot(N, hit.tangent);
        if (dot(Tt, Tt) > 1.0e-12) {
            Tt = normalize(Tt);
            float3 Bt = cross(N, Tt);
            float3 Np = normalize(Tt * nTex.x + Bt * nTex.y + N * nTex.z);
            coatN = dot(wo, Np) >= 0.0 ? Np : N;
            float3x3 coatTbn = buildTBN(coatN);
            coatT = coatTbn[0];
            coatB = coatTbn[1];
        }
    }
    if (mat.textureIndices2.z != kNoTexture) {
        float2 tDir = sampleBindless(mat.textureIndices2.z, hit.uv).xy * 2.0 - 1.0;
        float3 Tuv = hit.tangent - coatN * dot(coatN, hit.tangent);
        if (dot(Tuv, Tuv) > 1.0e-12 && dot(tDir, tDir) > 1.0e-12) {
            Tuv = normalize(Tuv);
            float3 Buv = cross(coatN, Tuv);
            float3 Tn = normalize(Tuv * tDir.x + Buv * tDir.y);
            float3 cT = Tn - coatN * dot(coatN, Tn);
            if (dot(cT, cT) > 1.0e-12) {
                coatT = normalize(cT);
                coatB = cross(coatN, coatT);
            }
        }
    }

    float3 emission = max(mat.emissionColorLum.xyz, 0.0.xxx) * max(mat.emissionColorLum.w, 0.0);
    if (max(emission.x, max(emission.y, emission.z)) > 0.0 && dot(wo, hit.geoNormal) > 0.0) {
        // Add emissive contribution only for:
        //   • primary camera ray (depth == 0) — camera directly sees the light
        //   • after a specular/delta bounce — NEE cannot sample through perfect mirrors
        // All other direct hits are handled by NEE below and must not be double-counted.
        if (pushConst.emissiveTriangleCount == 0u || payload.depth == 0u || payload.specularBounce) {
            payload.radiance += payload.throughput * emission;
        }
        payload.terminated = true;
        return;
    }

    // ── A1: Stochastic opacity gate (OpenPBR delta pass-through) ──
    // With probability (1-opacity), the ray passes straight through the surface with no 
    // interaction (delta BTDF per OpenPBR: geometry_opacity is the outermost lerp gate, not 
    // a per-lobe scale). Continue the ray in its original direction; otherwise shade normally.
    float opacity = saturate(mat.opacityFlagsPad.x);
    if (randFloat(payload.seed) < (1.0 - opacity)) {
        // Pass-through: continue ray in original direction, throughput unchanged (delta).
        payload.origin = hit.position + geoN * (dot(-normalize(WorldRayDirection()), geoN) >= 0.0 ? kRayEpsilon : -kRayEpsilon);
        payload.direction = -normalize(WorldRayDirection());
        payload.lastBsdfPdf = 1.0e30; // delta (needed for MIS weighting)
        payload.specularBounce = true;
        return;
    }

    uint seed = payload.seed;

    // --- Next Event Estimation: direct triangle area sampling ---
    // Ref: Shirley et al. "Monte Carlo Techniques for Direct Lighting Calculations" (1996)
    //      RTXPT — direct emissive triangle sampling with area-to-solid-angle PDF conversion
    if (pushConst.emissiveTriangleCount > 0u) {
        // Uniform triangle selection: pick index in [0, N).
        uint triIdx = min(uint(randFloat(seed) * float(pushConst.emissiveTriangleCount)),
                         pushConst.emissiveTriangleCount - 1u);
        GpuEmissiveTriangle tri = emissiveTriangles[triIdx];

        float3 v0     = tri.v0_area.xyz;
        float  area   = tri.v0_area.w;
        float3 edge1  = tri.edge1_emitR.xyz;
        float3 edge2  = tri.edge2_emitG.xyz;
        float3 lNorm  = tri.normal_emitB.xyz; // face normal (unit), outward from emitter
        float3 Le     = float3(tri.edge1_emitR.w, tri.edge2_emitG.w, tri.normal_emitB.w);

        // Uniform area sample via Shirley sqrt-folding.
        float  sqrtU  = sqrt(randFloat(seed));
        float  xi     = randFloat(seed);
        float3 bary   = float3(1.0 - sqrtU, sqrtU * (1.0 - xi), sqrtU * xi);
        float3 lPos   = v0 + edge1 * bary.y + edge2 * bary.z;

        float3 toLight  = lPos - hit.position;
        float  dist     = length(toLight);
        float3 wi       = toLight / max(dist, 1.0e-6);

        float cosLight = dot(-wi, lNorm); // positive when ray hits front face of emitter
        float cosI     = dot(wi, N);      // cosine at shading surface

        if (cosLight > 0.0 && cosI > 0.0) {
            // Area PDF to solid-angle PDF: pdf_ω = (1 / (N_tri × area)) × dist² / cosLight
            float areaPdf  = 1.0 / (float(pushConst.emissiveTriangleCount) * area);
            float solidPdf = areaPdf * dist * dist / cosLight;

            // Shadow ray stops just before the emitter surface to avoid self-intersection.
            // Subtract dot(geoN,wi)*kRayEpsilon to account for the shadow origin being offset
            // by geoN*kRayEpsilon — without this, surfaces whose geoN points toward the light
            // (e.g. a floor below a ceiling light) have their shadow ray reach the emitter at
            // exactly tMax, causing a false occlusion (floor always appears black).
            float3 shadowOrigin = hit.position + geoN * kRayEpsilon;
            float  shadowTMax   = max(dist - kRayEpsilon * (1.0 + dot(geoN, wi)), 0.0);
            if (traceNEEShadowRay(shadowOrigin, wi, shadowTMax)) {
                float3 bsdfVal = evalBSDF(mat, wo, wi, N, T, B, coatN, coatT, coatB);
                payload.radiance += payload.throughput * bsdfVal * Le * cosI / solidPdf;
            }
        }
    }

    // --- Next Event Estimation: env map importance sampling ---
    // Ref: PBR Book 4th ed §13.4.3 — MIS balance heuristic for env lights
    //      Veach "Robust Monte Carlo Methods" §9.2 — balance heuristic w = p_A / (p_A + p_B)
    if (pushConst.hasEnvMap != 0u && pushConst.envImportanceWidth > 0u) {
        float envPdf;
        float3 wi = sampleEnvImportance(seed, envPdf);
        float cosI = dot(wi, N);
        if (cosI > 0.0 && envPdf > 0.0) {
            float3 shadowOrigin = hit.position + geoN * kRayEpsilon;
            if (traceNEEShadowRay(shadowOrigin, wi, kInf)) {
                float3 Le = sampleEnvMap(wi) * pushConst.envLuminanceScale;
                float bsdfPdf = pdfBSDF(mat, wo, wi, N, T, B, coatN, coatT, coatB);
                float misW = envPdf / (envPdf + bsdfPdf); // balance heuristic
                payload.radiance += payload.throughput * evalBSDF(mat, wo, wi, N, T, B, coatN, coatT, coatB) * Le * cosI * misW / envPdf;
            }
        }
    }

    BSDFSample bsdfSample = sampleBSDF(mat, wo, N, T, B, coatN, coatT, coatB, seed);
    payload.seed = seed;
    if (bsdfSample.pdf <= 0.0 || max(bsdfSample.weight.x, max(bsdfSample.weight.y, bsdfSample.weight.z)) <= 0.0) {
        payload.terminated = true;
        return;
    }

    payload.specularBounce = bsdfSample.isDelta;
    payload.lastBsdfPdf    = bsdfSample.isDelta ? 1.0e30 : bsdfSample.pdf;
    payload.throughput *= bsdfSample.weight;
    payload.origin = hit.position + geoN * (dot(bsdfSample.direction, geoN) >= 0.0 ? kRayEpsilon : -kRayEpsilon);
    payload.direction = normalize(bsdfSample.direction);

    if (payload.depth >= kRRMinDepth) {
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
