import rng;
__exported import gpu_types;

/// Hyperion-specific per-instance GPU layout (std430, 32 bytes).
struct GpuInstance { uint meshIndex, materialIndex, vertexOffset, indexOffset, geometryKind; float sphereRadius; uint _pad[2]; };

struct CameraData {
    float4x4 invView;
    float4x4 invProj;
    float4   position;
    float    lensRadius;
    float    focusDistance;
    uint     frameIndex;
    uint     maxDepth;
    float    exposure;   ///< pre-computed: 1 / (1.2 * 2^EV100)
    float    _padCam[3];
};
struct PushConstants {
    uint  frameIndex;
    uint  maxDepth;
    uint  rngSeed;
    float envLuminanceScale;
    uint  lightCount;          ///< number of active GpuLights in lights buffer
    uint  outputColorSpace;    ///< OutputColorSpace enum (used by tonemap pass)
    uint  samplesPerPixel;     ///< samples accumulated per pixel this dispatch
    uint  hasEnvMap;           ///< 1 = IBL env map bound in set1/binding6, 0 = black background
    uint  emissiveTriangleCount; ///< number of emissive triangles for NEE area sampling (0 = disabled)
    uint  envImportanceWidth;    ///< CDF grid width for env importance sampling (0 = disabled)
    uint  envImportanceHeight;   ///< CDF grid height
    uint  tonemapper;            ///< Tonemapper enum (used by tonemap pass)
    uint  workingColorSpace;     ///< 0 = linear Rec.2020, 1 = linear Rec.709 (used by tonemap pass)
};

[raypayload]
struct RayPayload {
    float3 radiance      : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float3 throughput    : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float3 origin        : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float3 direction     : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    uint   depth         : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    RngState seed         : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    bool   terminated    : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    bool   specularBounce: read(caller, closesthit)       : write(caller, closesthit);
    // OpenPBR opacity pass-through: the vertex resolved to the ambient-medium side of
    // mix(ambient, surface, geometry_opacity), so no surface interaction happened and
    // the ray continued unchanged. raygen must not charge it against maxDepth — it is a
    // null interaction, exactly like a medium free-flight segment.
    bool   passedThrough : read(caller, closesthit)       : write(caller, closesthit);
    float  lastBsdfPdf   : read(caller, closesthit, miss) : write(caller, closesthit);
    // Step 3 (bulk subsurface random walk): when true, the NEXT TraceRay call is a segment
    // of a volumetric random walk rather than an ordinary bounce — raygen caps its TMax to
    // a freshly-sampled free-flight distance, a miss is a scattering event (miss.slang), and
    // a hit is a medium-boundary crossing (closesthit.slang's shadeMediumBoundary).
    bool   insideMedium   : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float3 mediumSigmaT   : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float3 mediumAlbedo   : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float  mediumG        : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float  mediumEta      : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float  mediumFreeDist : read(caller, closesthit, miss) : write(caller, closesthit, miss);
};

/// Shadow-ray payload. `transmittance` is the running ∏(1-α) accumulated by the
/// shadow any-hit over cut-out crossings; `missed` is set only by the shadow
/// miss shader, so it distinguishes "the ray got through (possibly attenuated)"
/// from "a fully present surface accepted the hit and ended the search" — the
/// latter also covers hits traversal accepted on its own (an opacity micromap's
/// OPAQUE microtriangles never invoke the any-hit).
[raypayload]
struct ShadowPayload {
    bool  missed        : read(caller, miss)         : write(caller, miss);
    float transmittance : read(caller, anyhit, miss) : write(caller, anyhit);
};

struct HitInfo {
    float3 position;
    float3 smoothPosition; ///< shadow/NEE ray origin: hit point lifted onto the smooth (shading-normal) surface
    float3 normal;
    float3 geoNormal;
    float3 tangent;   ///< UV-aligned surface tangent (world space); used for normal mapping
    float2 uv;
    uint   instanceIndex;
    uint   primitiveIndex;
    bool   frontFace;
};

struct SphereIntersectionAttributes {
    float t;
};

[[vk::binding(0, 0)]] uniform RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> hdrTarget;
[[vk::binding(2, 0)]] ConstantBuffer<CameraData> camera;
[[vk::binding(3, 0)]] RWTexture2D<float4> swapchainOutput;
[[vk::binding(4, 0)]] RWTexture2D<float4> gNormal;  ///< G-buffer: world-space normal (xyz), w=0
[[vk::binding(5, 0)]] RWTexture2D<float>  gDepth;   ///< G-buffer: ray hit distance (kInf = sky)

[[vk::binding(0, 1)]] StructuredBuffer<GpuInstance>  instances;
[[vk::binding(1, 1)]] StructuredBuffer<GpuMaterial>  materials;
[[vk::binding(2, 1)]] StructuredBuffer<GpuVertex>    vertices;
[[vk::binding(3, 1)]] StructuredBuffer<uint>         indices;
[[vk::combinedImageSampler]] [[vk::binding(4, 1)]] Texture2D<float4> bindlessTextures[1024];
[[vk::combinedImageSampler]] [[vk::binding(4, 1)]] SamplerState      bindlessSamplers[1024];
[[vk::binding(5, 1)]] StructuredBuffer<GpuLight>     lights;

/// IBL equirectangular env map (binding 6, set 1).  Only sampled when pushConst.hasEnvMap != 0.
[[vk::combinedImageSampler]] [[vk::binding(6, 1)]] Texture2D<float4> envMap;
[[vk::combinedImageSampler]] [[vk::binding(6, 1)]] SamplerState      envSampler;

/// Emissive triangle buffer for NEE direct area sampling (binding 7, set 1).
[[vk::binding(7, 1)]] StructuredBuffer<GpuEmissiveTriangle> emissiveTriangles;

/// 2D separable CDF buffers for env map importance sampling (bindings 8–9, set 1).
/// Ref: PBR Book 4th ed §12.5 — separable CDF for infinite area lights
/// Both bindings are PARTIALLY_BOUND; only valid when pushConst.envImportanceWidth > 0.
[[vk::binding(8, 1)]] StructuredBuffer<float> envMarginalCdf;     ///< (H+1) normalized row CDFs
[[vk::binding(9, 1)]] StructuredBuffer<float> envConditionalCdf;  ///< H×(W+1) per-row column CDFs

/// Power-proportional selection CDF for emissive-triangle NEE (binding 10, set 1).
/// cdf[i] = (Σ_{j≤i} area_j·luminance(Le_j)) / totalPower, so cdf[N-1] == 1.
[[vk::binding(10, 1)]] StructuredBuffer<float> emissiveCdf;

[[vk::push_constant]] PushConstants pushConst;
