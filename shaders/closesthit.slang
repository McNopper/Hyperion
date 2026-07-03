import common;
import math;
import bsdf;
import env;
import path_integrator;

#ifdef POSITION_FETCH_ENABLED
/// Fetch the object-space position of a hit-triangle vertex via
/// VK_KHR_ray_tracing_position_fetch.  Normals/UVs/tangents still come from
/// the vertex SSBO; only positions use this built-in.
float3 getHitTriangleVertexPosition(int index)
{
    return spirv_asm {
        OpCapability RayTracingPositionFetchKHR;
        OpExtension "SPV_KHR_ray_tracing_position_fetch";
        %ptr : $$Ptr<float3, Access.Read, AddressSpace.VaryingInput> = OpAccessChain builtin(HitTriangleVertexPositionsKHR:float3[3]) $index;
        result:$$float3 = OpLoad %ptr;
    };
}
#endif

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

#ifdef POSITION_FETCH_ENABLED
    float3 p0 = getHitTriangleVertexPosition(0);
    float3 p1 = getHitTriangleVertexPosition(1);
    float3 p2 = getHitTriangleVertexPosition(2);
#else
    float3 p0 = v0.position;
    float3 p1 = v1.position;
    float3 p2 = v2.position;
#endif

    float b1 = attr.barycentrics.x;
    float b2 = attr.barycentrics.y;
    float b0 = 1.0 - b1 - b2;

    hit.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    hit.geoNormal = normalize(cross(p1 - p0, p2 - p0));
    hit.normal = normalize(v0.normal * b0 + v1.normal * b1 + v2.normal * b2);
    if (dot(hit.normal, hit.geoNormal) < 0.0) {
        hit.normal = -hit.normal;
    }
    hit.uv = v0.uv * b0 + v1.uv * b1 + v2.uv * b2;

    // UV-aligned tangent from triangle position/UV deltas (Lengyel's method).
    // Computed per-hit so normal mapping does not depend on baked vertex tangents.
    float2 duv1 = v1.uv - v0.uv;
    float2 duv2 = v2.uv - v0.uv;
    float3 dp1  = p1 - p0;
    float3 dp2  = p2 - p0;
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

/// Shadow-ray tracer for the shared path integrator (ITracer).
/// Returns true when origin→direction*tMax is unoccluded (the shadow miss shader fired).
/// All instances are visible (kInstanceMaskAll); tMax stops just before the sampled light
/// surface so the ray does not self-intersect it.
struct HyperionTracer : ITracer {
    bool traceOccluded(float3 origin, float3 direction, float tMax) {
        ShadowPayload p;
        p.inShadow = true;
        RayDesc ray;
        ray.Origin    = origin;
        ray.Direction = direction;
        ray.TMin      = 0.0;
        ray.TMax      = tMax;
        TraceRay(tlas,
                 RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                 kInstanceMaskAll, 0, 2, 1, ray, p);
        return !p.inShadow;
    }
}

/// Step 3 (bulk subsurface random walk): handle a ray that struck geometry while inside the
/// medium — a boundary crossing. Stochastically exits (Fresnel transmittance) or totally
/// internally reflects (continues the walk), using the SAME dielectric-interface convention
/// as the specular/transmission lobes (bsdf_shared `fresnelDielectric`, the material's
/// `specular_ior` carried in `payload.mediumEta`). Uses the GEOMETRIC normal (interpolated
/// shading normals can fake a wrong-sided interface on smooth-shaded meshes). hit.geoNormal
/// points OUTWARD from OBJ winding regardless of which side the ray struck.
void shadeMediumBoundary(inout RayPayload payload, HitInfo hit)
{
    float3 rayDir = normalize(WorldRayDirection());
    float3 outwardN = hit.geoNormal;

    // G5/G6 spectral boundary factor: the ray reached the boundary at t_hit before the sampled
    // flight distance. Apply the per-channel transmittance to the boundary divided by the
    // hero-wavelength "reach the boundary" pdf (balance heuristic):
    //   T_b = exp(-σ_t·t_hit),  pdf_surf = (1/3)·Σ_c T_b_c.
    // For achromatic σ_t this is exactly 1 (the old scalar walk applied no boundary factor).
    {
        float3 st = max(payload.mediumSigmaT, 1.0e-6.xxx);
        float3 Tb = exp(-st * RayTCurrent());
        float pdfSurf = (Tb.x + Tb.y + Tb.z) * (1.0 / 3.0);
        payload.throughput *= Tb / max(pdfSurf, 1.0e-8);
    }

    // The ray is INSIDE the medium travelling toward the boundary, so this is a medium→air
    // crossing. fresnelDielectric selects the medium→air branch (ei=eta, et=1, and reports
    // TIR when the angle exceeds the critical angle) only for a NEGATIVE cosine, so pass the
    // negated incidence cosine — passing abs() would wrongly evaluate the air→medium branch
    // and never report total internal reflection.
    float cosI = abs(dot(rayDir, outwardN));
    float F = fresnelDielectric(-cosI, payload.mediumEta);

    // Refract to exit; refract() returns 0 on total internal reflection. F==1 in that case, so
    // the reflect branch below is taken with probability 1 — but guard the exit branch too so a
    // near-critical numerical miss reflects and STAYS inside rather than leaking out unrefracted.
    float3 refracted = refract(rayDir, -outwardN, payload.mediumEta);
    bool tir = dot(refracted, refracted) < 1.0e-9;

    if (tir || randFloat(payload.seed) < F) {
        // (Total) internal reflection — chosen with probability F (=1 under TIR); the stochastic
        // branch selection carries the Fresnel weight, so no throughput scaling is applied.
        payload.direction = reflect(rayDir, outwardN);
        payload.origin = hit.position - outwardN * kMediumEpsilon;
        return;
    }

    // Exit into the exterior (air, eta=1) and resume ordinary shading on the next bounce. The
    // (1-F) transmittance is already accounted for by the stochastic branch selection.
    payload.direction = normalize(refracted);
    payload.insideMedium = false;
    payload.origin = hit.position + outwardN * kMediumEpsilon;
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

    // ── Shared surface estimator ─────────────────────────────────────────────────
    // Emission guard, stochastic opacity gate, emissive-triangle + environment NEE,
    // BSDF continuation and Russian roulette all live in Harmonia's path_integrator
    // module, parameterized over ITracer so Hyperion (TraceRay) and Theia (RayQuery)
    // share the identical unidirectional Monte-Carlo estimator.
    SurfaceHit surf;
    surf.position  = hit.position;
    surf.normal    = N;
    surf.geoNormal = geoN;
    surf.T = T; surf.B = B;
    surf.coatN = coatN; surf.coatT = coatT; surf.coatB = coatB;
    surf.mat = mat;
    surf.wo  = wo;

    PathState state;
    state.radiance       = payload.radiance;
    state.throughput     = payload.throughput;
    state.depth          = payload.depth;
    state.seed           = payload.seed;
    state.terminated     = false;
    state.specularBounce = payload.specularBounce;
    state.lastBsdfPdf    = payload.lastBsdfPdf;
    state.nextOrigin     = payload.origin;
    state.nextDirection  = payload.direction;
    state.insideMedium   = payload.insideMedium;
    state.mediumSigmaT   = payload.mediumSigmaT;
    state.mediumAlbedo   = payload.mediumAlbedo;
    state.mediumG        = payload.mediumG;
    state.mediumEta      = payload.mediumEta;

    HyperionTracer tracer;
    integrateSurface(tracer, surf,
        pushConst.emissiveTriangleCount, emissiveTriangles, emissiveCdf,
        envMarginalCdf, envConditionalCdf,
        pushConst.envImportanceWidth, pushConst.envImportanceHeight,
        (pushConst.hasEnvMap != 0u), pushConst.envLuminanceScale,
        envMap, envSampler,
        state);

    payload.radiance       = state.radiance;
    payload.throughput     = state.throughput;
    payload.seed           = state.seed;
    payload.terminated     = state.terminated;
    payload.specularBounce = state.specularBounce;
    payload.lastBsdfPdf    = state.lastBsdfPdf;
    payload.origin         = state.nextOrigin;
    payload.direction      = state.nextDirection;
    payload.insideMedium   = state.insideMedium;
    payload.mediumSigmaT   = state.mediumSigmaT;
    payload.mediumAlbedo   = state.mediumAlbedo;
    payload.mediumG        = state.mediumG;
    payload.mediumEta      = state.mediumEta;
}

[shader("closesthit")]
void triangleClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    HitInfo hit = makeTriangleHit(attr);
    if (payload.insideMedium) {
        shadeMediumBoundary(payload, hit);
        return;
    }
    shadeSurface(payload, hit);
}

[shader("closesthit")]
void sphereClosestHit(inout RayPayload payload, in SphereIntersectionAttributes attr)
{
    HitInfo hit = makeSphereHit(attr);
    if (payload.insideMedium) {
        shadeMediumBoundary(payload, hit);
        return;
    }
    shadeSurface(payload, hit);
}
