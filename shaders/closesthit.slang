import common;
import math;
import bsdf;
import env;
import path_integrator;
import medium_walk;
import surface;

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

    // Vertices live in object space; the TLAS instance transform places them. Fetch
    // the instance matrix and its inverse-transpose (for normals) from the builtins.
    float3x4 o2w = ObjectToWorld3x4();
    float3x3 normalMat = transpose((float3x3)WorldToObject3x4());

#ifdef POSITION_FETCH_ENABLED
    // Hardware-supplied triangle positions are OBJECT-space (Vulkan spec,
    // VK_KHR_ray_tracing_position_fetch: HitTriangleVertexPositionsKHR) — transform
    // by the instance matrix exactly like the SSBO path below. Treating them as
    // world-space made geoNormal/tangent object-space on every TLAS-rotated
    // instance, silently desynchronizing them from the correctly-transformed
    // shading normal (the long-standing self-shadow artifact).
    float3 p0 = mul(o2w, float4(getHitTriangleVertexPosition(0), 1.0));
    float3 p1 = mul(o2w, float4(getHitTriangleVertexPosition(1), 1.0));
    float3 p2 = mul(o2w, float4(getHitTriangleVertexPosition(2), 1.0));
#else
    float3 p0 = mul(o2w, float4(v0.position, 1.0));
    float3 p1 = mul(o2w, float4(v1.position, 1.0));
    float3 p2 = mul(o2w, float4(v2.position, 1.0));
#endif

    // Normals come from the SSBO in object space — transform by the inverse-transpose.
    float3 n0 = normalize(mul(normalMat, v0.normal));
    float3 n1 = normalize(mul(normalMat, v1.normal));
    float3 n2 = normalize(mul(normalMat, v2.normal));

    float b1 = attr.barycentrics.x;
    float b2 = attr.barycentrics.y;
    float b0 = 1.0 - b1 - b2;

    hit.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    // Shared geometry reconstruction (geoNormal / shading normal / Lengyel tangent / UV).
    TriangleGeometry g = reconstructTriangleGeometry(p0, p1, p2, n0, n1, n2, v0.uv, v1.uv, v2.uv,
                                                      float3(b0, b1, b2));
    hit.geoNormal = g.geoNormal;
    hit.normal    = g.shadingNormal;

    // Shadow-terminator handling is the Chiang 2019 shadowing factor, applied by the
    // shared estimator (see Harmonia bsdf_shared.slang for the scheme + why the
    // Hanika origin lift was rejected: its magnitude scales with triangle size).
    // smoothPosition stays == position (hook kept for renderer-specific schemes).
    hit.smoothPosition = hit.position;
    hit.uv = g.uv;
    hit.tangent = g.tangent;

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

    // Analytic sphere mesh is centred at the object-space origin; its world centre is
    // the instance translation (ObjectToWorld applied to the origin).
    float3x4 o2w = ObjectToWorld3x4();
    float3 center = mul(o2w, float4(0.0, 0.0, 0.0, 1.0));

    hit.position = WorldRayOrigin() + WorldRayDirection() * attr.t;
    hit.geoNormal = normalize(hit.position - center);
    hit.normal = hit.geoNormal;
    hit.smoothPosition = hit.position; // analytic sphere: exact normals, no terminator lift needed

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

/// Bindless-texture fetcher for the shared surface-frame populator (mip 0).
struct HyperionTextureSampler : ISurfaceTextureSampler {
    float4 sampleTex(uint idx, float2 uv) {
        return bindlessTextures[NonUniformResourceIndex(idx)]
            .SampleLevel(bindlessSamplers[NonUniformResourceIndex(idx)], uv, 0.0);
    }
};

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
/// medium — a boundary crossing. All walk math (transmittance, Fresnel exit/TIR) lives in the
/// shared `medium_walk` module; this thin shim just maps RayPayload <-> the helpers. Uses the
/// GEOMETRIC normal (interpolated shading normals can fake a wrong-sided interface); hit.geoNormal
/// is the RAW outward-winding normal (dielectric-sidedness contract).
void shadeMediumBoundary(inout RayPayload payload, HitInfo hit)
{
    float3 outwardN = hit.geoNormal;

    mediumBoundaryTransmittance(payload.throughput, payload.mediumSigmaT,
                                mediumIsPureAbsorber(payload.mediumAlbedo), RayTCurrent());

    float3 outOrigin, outDir;
    bool exited = mediumResolveExit(payload.seed, payload.mediumEta,
                                    WorldRayDirection(), outwardN, hit.position,
                                    outOrigin, outDir);
    payload.direction = outDir;
    payload.origin    = outOrigin;
    if (exited) {
        payload.insideMedium = false;
    }
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
    float3 wo = -normalize(WorldRayDirection());

    // ── Texture maps + shading frame (shared OpenPBR populator) ─────────────────
    // Base/ORM/emission unpack, normal mapping, geometry_tangent anisotropy reorientation
    // and the coat frame all live in Harmonia's `surface` module (shared 1:1 with Theia).
    HyperionTextureSampler tex;
    SurfaceFrame f = applyOpenPbrTextureMaps(mat, tex, wo, hit.normal, hit.geoNormal, hit.tangent, hit.uv);

    // ── Shared surface estimator ─────────────────────────────────────────────────
    // Emission guard, stochastic opacity gate, emissive-triangle + environment NEE,
    // BSDF continuation and Russian roulette all live in Harmonia's path_integrator
    // module, parameterized over ITracer so Hyperion (TraceRay) and Theia (RayQuery)
    // share the identical unidirectional Monte-Carlo estimator.
    SurfaceHit surf;
    surf.position  = hit.position;
    surf.shadowPos = hit.smoothPosition;
    surf.normal    = f.normal;
    surf.geoNormal = f.geoNormal;
    surf.backface  = f.backface;
    surf.T = f.T; surf.B = f.B;
    surf.coatN = f.coatN; surf.coatT = f.coatT; surf.coatB = f.coatB;
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
