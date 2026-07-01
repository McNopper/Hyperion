import rng;

struct GpuVertex   { float3 position; float tangentX; float3 normal; float tangentY; float2 uv; float tangentZ; float bitangentSign; };
struct GpuMaterial {
    float4 baseColorWeight;
    float4 baseMetalnessDiffRough;
    float4 specularColorWeight;
    float4 specularRoughAnisoIor;
    float4 transmissionColorWeight;
    float4 transmissionParams;     ///< x = transmission_depth, z = dispersion_scale, w = dispersion_abbe_number
    float4 transmissionScatter;    ///< xyz = transmission_scatter albedo, w = transmission_scatter_anisotropy (g)
    float4 subsurfaceColorWeight;
    float4 subsurfaceRadiusScale;
    uint4  textureIndices;  ///< bindless indices: [base_color, normal, orm, emission]; ~0u = none
    float4 thinFilmParams;
    float4 coatColorWeight;
    float4 coatRoughAnisoIorDark;
    float4 fuzzColorWeight;
    float4 fuzzRoughPad;
    float4 emissionColorLum;
    float4 opacityFlagsPad;        ///< x = opacity, y = flags, z = subsurface_scatter_anisotropy, w = thin_walled
    uint4  textureIndices2; ///< bindless indices: [coat_normal, tangent, coat_tangent, unused]; ~0u = none
};
struct GpuInstance { uint meshIndex, materialIndex, vertexOffset, indexOffset, geometryKind; float sphereRadius; uint _pad[2]; };

/// Per-triangle emissive descriptor for NEE direct area sampling (std430, 64 bytes = 4×float4).
/// Edge vectors and emission components share float4 w-channels to avoid padding.
struct GpuEmissiveTriangle {
    float4 v0_area;       ///< xyz = v0 world pos, w = triangle area
    float4 edge1_emitR;   ///< xyz = edge1 (v1-v0) world, w = emission.r
    float4 edge2_emitG;   ///< xyz = edge2 (v2-v0) world, w = emission.g
    float4 normal_emitB;  ///< xyz = face normal (unit) world, w = emission.b
};

/// Light type constants — match LightType C++ enum.
static const uint kLightRect        = 0;
static const uint kLightPoint       = 1;
static const uint kLightSpot        = 2;
static const uint kLightDirectional = 3;
static const uint kLightSky         = 4;

/// GPU light descriptor.  intensity is in radiometric units (see Light.cpp toGpu()).
struct GpuLight {
    float3 position;   float  typeF;      ///< asuint(typeF) == LightType constant
    float3 direction;  float  range;      ///< attenuation cutoff; 0 = infinite
    float3 color;      float  intensity;  ///< radiometric (W/sr, W/sr/m², or W/m²)
    float  halfWidth;
    float  halfHeight;
    float  cosInner;
    float  cosOuter;
};

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

[raypayload]
struct ShadowPayload {
    bool inShadow : read(caller, miss) : write(caller, miss);
};

struct HitInfo {
    float3 position;
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

static const float kPi = 3.14159265359;
static const float kInvPi = 0.31830988618;
static const float kHalfPi = 1.57079632679; ///< π/2
static const float kRayEpsilon = 0.001;
/// Self-intersection offset for volumetric random-walk segments (Step 3/4). Much smaller than
/// kRayEpsilon because a dense medium's mean free path can be far below 0.001, where the
/// surface-scale epsilon would overshoot the next scatter distance and leak through boundaries.
static const float kMediumEpsilon = 1.0e-5;
static const float kInf = 1.0e30;

/// Rec.2020 luminance coefficients (ITU-R BT.2020).
static const float3 kRec2020Luma = float3(0.2627, 0.6780, 0.0593);

/// TLAS instance mask bit for TraceRay culling.
/// Every TraceRay call uses kInstanceMaskAll as its InstanceInclusionMask (all instances visible).
static const uint kInstanceMaskAll = 0xFF; ///< all instances visible

/// Minimum path depth before Russian Roulette path termination is applied.
static const uint kRRMinDepth = 3u;

/// Firefly suppression: maximum per-sample luminance after exposure (display units).
static const float kFireflyClamp = 100.0;

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
