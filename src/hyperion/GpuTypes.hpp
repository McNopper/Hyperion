#pragma once

#include <volk/volk.h>

#include <cstdint>
#include <type_traits>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

/// Scene light types (matches GpuLight::type field and shader constants).
enum class LightType : uint32_t {
    Rect = 0,        ///< Area / rectangular emitter  — intensity in cd/m² (nits)
    Point = 1,       ///< Omnidirectional point light  — intensity in cd or lm
    Spot = 2,        ///< Cone spot light              — intensity in cd or lm
    Directional = 3, ///< Infinitely distant parallel  — intensity in lux
    Sky = 4,         ///< IBL sky dome                 — intensity in cd/m²
};

struct GpuVertex {
    glm::vec3 position;
    float tangentX; ///< Tangent vector X component (world space)
    glm::vec3 normal;
    float tangentY; ///< Tangent vector Y component (world space)
    glm::vec2 uv;
    float tangentZ;      ///< Tangent vector Z component (world space)
    float bitangentSign; ///< ±1 handedness of the bitangent (B = sign × (N × T))
};

struct GpuMaterial {
    glm::vec4 baseColorWeight;
    glm::vec4 baseMetalnessDiffRough;
    glm::vec4 specularColorWeight;
    glm::vec4 specularRoughAnisoIor;
    glm::vec4 transmissionColorWeight;
    glm::vec4 transmissionParams;
    glm::vec4 transmissionScatter;
    glm::vec4 subsurfaceColorWeight;
    glm::vec4 subsurfaceRadiusScale;
    glm::uvec4 textureIndices; ///< bindless texture indices: [base_color, normal, orm, emission]; ~0u = none
    glm::vec4 thinFilmParams;
    glm::vec4 coatColorWeight;
    glm::vec4 coatRoughAnisoIorDark;
    glm::vec4 fuzzColorWeight;
    glm::vec4 fuzzRoughPad;
    glm::vec4
        emissionColorLum; ///< xyz = emission_color (linear Rec.2020), w = emission_luminance in cd/m² (OpenPBR spec)
    glm::vec4 opacityFlagsPad;
};

struct GpuInstance {
    uint32_t meshIndex;
    uint32_t materialIndex;
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t geometryKind;
    float sphereRadius;
    uint32_t _pad[2];
};

/// Per-triangle emissive descriptor for NEE direct area sampling (std430, 64 bytes = 4×float4).
/// Edge vectors and emission components share float4 w-channels to avoid padding.
struct GpuEmissiveTriangle {
    glm::vec4 v0_area;      ///< xyz = v0 world pos, w = triangle area
    glm::vec4 edge1_emitR;  ///< xyz = edge1 (v1-v0) world, w = emission.r
    glm::vec4 edge2_emitG;  ///< xyz = edge2 (v2-v0) world, w = emission.g
    glm::vec4 normal_emitB; ///< xyz = face normal (unit) world, w = emission.b
};

/// GPU-side light descriptor (std430, 64 bytes).
///
/// `intensity` is stored in radiometric units after photometric → radiometric
/// conversion in Light::toGpu() (divides by the luminous efficacy constant 683 lm/W):
///   Rect / Sky   : radiant exitance  [W/sr/m²]  = luminance [cd/m²]    / 683
///   Point / Spot : radiant intensity [W/sr]      = luminous intensity [cd] / 683
///   Directional  : irradiance        [W/m²]      = illuminance [lux]     / 683
struct GpuLight {
    glm::vec3 position;
    float type; ///< reinterpret_cast<uint32_t> → LightType
    glm::vec3 direction;
    float range; ///< attenuation cutoff; 0 = infinite
    glm::vec3 color;
    float intensity;  ///< radiometric (see above)
    float halfWidth;  ///< rect half-width  / spot unused
    float halfHeight; ///< rect half-height / spot unused
    float cosInner;   ///< spot inner cone cos(angle)
    float cosOuter;   ///< spot outer cone cos(angle)
};

struct CameraData {
    glm::mat4 invView;
    glm::mat4 invProj;
    glm::vec4 position;
    float lensRadius;
    float focusDistance;
    uint32_t frameIndex;
    uint32_t maxDepth;
    float exposure; ///< pre-computed from EV100: 1 / (1.2 * 2^EV100)
    float _padCam[3];
};

struct PushConstants {
    uint32_t frameIndex;
    uint32_t maxDepth;
    uint32_t rngSeed;
    float envLuminanceScale;
    uint32_t lightCount;         ///< number of active GpuLights in the light buffer
    uint32_t outputColorSpace;   ///< OutputColorSpace enum value (used by tonemap pass)
    uint32_t samplesPerPixel;    ///< samples per pixel this dispatch
    uint32_t hasEnvMap;          ///< 1 = IBL env map is bound in set1/binding6, 0 = procedural sky
    uint32_t emissiveTriangleCount; ///< number of emissive triangles for NEE area sampling (0 = disabled)
    uint32_t envImportanceWidth;    ///< CDF grid width for env importance sampling (0 = disabled)
    uint32_t envImportanceHeight;   ///< CDF grid height for env importance sampling
    uint32_t _pad;               // NOLINT(modernize-avoid-c-arrays) — explicit GPU layout padding
};

/// TLAS instance mask bit used in TraceRay InstanceInclusionMask comparisons.
/// All instances use the default mask (all-bits-set); no per-type masking is needed
/// since shadow rays stop just before the emissive surface via a calibrated tMax.
static constexpr uint32_t kInstanceMaskAll = 0xFFU; ///< all instances visible

static_assert(std::is_trivially_copyable_v<GpuVertex>);

/// Sentinel texture index: slot holds no texture.
static constexpr uint32_t kNoTexture = ~0u;
static_assert(std::is_trivially_copyable_v<GpuMaterial>);
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(std::is_trivially_copyable_v<GpuLight>);
static_assert(std::is_trivially_copyable_v<GpuEmissiveTriangle>);
static_assert(std::is_trivially_copyable_v<CameraData>);
static_assert(std::is_trivially_copyable_v<PushConstants>);

static_assert(sizeof(GpuVertex) == 48);
static_assert(sizeof(GpuMaterial) == 272);
static_assert(sizeof(GpuInstance) == 32);
static_assert(sizeof(GpuLight) == 64);
static_assert(sizeof(GpuEmissiveTriangle) == 64);
static_assert(sizeof(CameraData) == 176);
static_assert(sizeof(PushConstants) == 48);
