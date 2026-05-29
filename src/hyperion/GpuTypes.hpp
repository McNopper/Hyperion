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

/// GPU-side emissive-mesh descriptor for NEE bounding-sphere sampling (std430, 32 bytes).
struct GpuEmissiveLight {
    glm::vec3 center;       ///< world-space bounding sphere centre (= mesh centroid)
    float radius;           ///< bounding sphere radius
    glm::vec3 emission;     ///< radiance: emissionColor × emissionLuminance (linear Rec.2020)
    uint32_t instanceIndex; ///< instance index used to skip self-shadow during NEE
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
    uint32_t emissiveLightCount; ///< number of emissive mesh lights for NEE (0 = disabled)
    uint32_t _pad[3];            // NOLINT(modernize-avoid-c-arrays) — explicit GPU layout padding
};

/// TLAS instance mask bits used in TraceRay InstanceInclusionMask comparisons.
/// An instance is tested by a ray when (instance.mask & ray.cullMask) != 0.
static constexpr uint32_t kInstanceMaskAll = 0xFFU;      ///< all instances visible
static constexpr uint32_t kInstanceMaskEmissive = 0x02U; ///< bit 1: emissive mesh lights
static constexpr uint32_t kShadowRayMask = kInstanceMaskAll & ~kInstanceMaskEmissive;

static_assert(std::is_trivially_copyable_v<GpuVertex>);

/// Sentinel texture index: slot holds no texture.
static constexpr uint32_t kNoTexture = ~0u;
static_assert(std::is_trivially_copyable_v<GpuMaterial>);
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(std::is_trivially_copyable_v<GpuLight>);
static_assert(std::is_trivially_copyable_v<GpuEmissiveLight>);
static_assert(std::is_trivially_copyable_v<CameraData>);
static_assert(std::is_trivially_copyable_v<PushConstants>);

static_assert(sizeof(GpuVertex) == 48);
static_assert(sizeof(GpuMaterial) == 272);
static_assert(sizeof(GpuInstance) == 32);
static_assert(sizeof(GpuLight) == 64);
static_assert(sizeof(GpuEmissiveLight) == 32);
static_assert(sizeof(CameraData) == 176);
static_assert(sizeof(PushConstants) == 48);
