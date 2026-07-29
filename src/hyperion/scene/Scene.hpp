#ifndef HYPERION_SCENE_SCENE_HPP
#define HYPERION_SCENE_SCENE_HPP

#include <volk/volk.h>

#include <cstdint>
#include <string_view>
#include <vector>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/CommandPool.hpp"
#include "harmonia/renderer/AccelerationStructure.hpp"
#include "harmonia/scene/Geometry.hpp"
#include "harmonia/scene/ISceneBuilder.hpp"
#include "harmonia/scene/Light.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/SceneBase.hpp"
#include "harmonia/scene/Texture.hpp"

/// Hyperion (path tracer) per-instance GPU layout (std430, 32 bytes). The path tracer
/// fetches triangles via an index buffer, so this carries the mesh's index/vertex
/// range. Distinct from Theia's rasterizer layout, which carries meshlet ranges.
/// Object→world placement comes from the TLAS instance transform (read in-shader via
/// ObjectToWorld3x4), so no matrix is stored here.
struct GpuInstance {
    std::uint32_t meshIndex = 0; ///< references the unique mesh (for debugging)
    std::uint32_t materialIndex = 0;
    std::uint32_t vertexOffset = 0; ///< mesh's first vertex in the global vertex buffer
    std::uint32_t indexOffset = 0;  ///< mesh's first index in the global index buffer
    std::uint32_t geometryKind = 0; ///< 0 = triangle mesh, 1 = analytic sphere
    float sphereRadius = 0.0f;
    std::uint32_t _pad[2] = {};
};
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(sizeof(GpuInstance) == 32);

class Scene : public harmonia::SceneBase {
  public:
    // addMaterial / addTexture / addInstance / addMesh / build are inherited concrete from SceneBase.

    [[nodiscard]] std::uint32_t
    addSphereMesh(const DeviceContext& ctx, const CommandPool& pool, float radius, std::string_view name = "") override;

    [[nodiscard]] VkAccelerationStructureKHR tlas() const noexcept { return m_tlas.handle(); }
    [[nodiscard]] VkDeviceAddress tlasAddress() const noexcept { return m_tlasAddress; }
    [[nodiscard]] const Buffer& instanceBuffer() const noexcept { return m_instanceBuffer; }
    [[nodiscard]] const Buffer& materialBuffer() const noexcept { return m_materialBuffer; }
    [[nodiscard]] const Buffer& vertexBuffer() const noexcept { return m_vertexBuffer; }
    [[nodiscard]] const Buffer& indexBuffer() const noexcept { return m_indexBuffer; }
    [[nodiscard]] const Buffer& lightBuffer() const noexcept { return m_lightBuffer; }
    [[nodiscard]] const Buffer& emissiveTriangleBuffer() const noexcept { return m_emissiveTriangleBuffer; }
    [[nodiscard]] const Buffer& emissiveCdfBuffer() const noexcept { return m_emissiveCdfBuffer; }
    [[nodiscard]] std::uint32_t instanceCount() const noexcept {
        return static_cast<std::uint32_t>(m_instances.size());
    }
    [[nodiscard]] std::uint32_t lightCount() const noexcept { return static_cast<std::uint32_t>(m_lights.size()); }
    [[nodiscard]] std::uint32_t emissiveTriangleCount() const noexcept { return m_emissiveTriangleCount; }

  private:
    /// Per-mesh GPU layout computed in buildSceneBuffers (parallel to m_meshes).
    struct MeshGpu {
        std::uint32_t vertexOffset = 0;
        std::uint32_t indexOffset = 0;
        std::uint32_t geometryKind = 0; ///< 0 = triangle mesh, 1 = analytic sphere
        float sphereRadius = 0.0f;
    };

    VkResult buildSceneBuffers(const DeviceContext& ctx, const CommandPool& pool) override;

    void buildGpuInstances();
    VkResult uploadAllBuffers(const DeviceContext& ctx,
                              const CommandPool& pool,
                              const std::vector<GpuMaterial>& gpuMaterials,
                              const std::vector<GpuVertex>& vertices,
                              const std::vector<std::uint32_t>& indices,
                              const std::vector<GpuLight>& gpuLights,
                              const std::vector<GpuEmissiveTriangle>& emissiveTriangles,
                              const std::vector<float>& emissiveCdf);

    std::vector<MeshGpu> m_meshGpu;          ///< per-mesh GPU ranges
    std::vector<GpuInstance> m_gpuInstances; ///< per-instance GPU rows (built at build)
    Buffer m_instanceBuffer{};
    Buffer m_materialBuffer{};
    Buffer m_vertexBuffer{};
    Buffer m_indexBuffer{};
    Buffer m_lightBuffer{};
    Buffer m_emissiveTriangleBuffer{};
    Buffer m_emissiveCdfBuffer{};
    std::uint32_t m_emissiveTriangleCount = 0;
};
#endif // HYPERION_SCENE_SCENE_HPP
