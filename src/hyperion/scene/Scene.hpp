#pragma once

#include <volk/volk.h>

#include <expected>
#include <memory>
#include <string>
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
/// fetches triangles via an index buffer, so this carries indexOffset. Distinct from
/// Theia's rasterizer layout, which carries meshletOffset/meshletCount instead.
struct GpuInstance {
    uint32_t meshIndex     = 0;
    uint32_t materialIndex = 0;
    uint32_t vertexOffset  = 0;
    uint32_t indexOffset   = 0;
    uint32_t geometryKind  = 0;
    float    sphereRadius  = 0.0f;
    uint32_t _pad[2]       = {};
};
static_assert(std::is_trivially_copyable_v<GpuInstance>);
static_assert(sizeof(GpuInstance) == 32);

class SceneBuilder;

class Scene : public harmonia::SceneBase {
  public:
    using Builder = SceneBuilder;

    [[nodiscard]] uint32_t addMaterial(Material&& mat) override { return harmonia::SceneBase::addMaterial(std::move(mat)); }
    [[nodiscard]] uint32_t addTexture(Texture&& texture) override { return harmonia::SceneBase::addTexture(std::move(texture)); }

    [[nodiscard]] uint32_t addMesh(const DeviceContext& ctx,
                                   const CommandPool& pool,
                                   MeshData&& data,
                                   uint32_t materialIdx,
                                   std::string_view name = "") override;
    [[nodiscard]] uint32_t addSphere(const DeviceContext& ctx,
                                     const CommandPool& pool,
                                     sm::float3 center,
                                     float radius,
                                     uint32_t materialIdx) override;

    VkResult build(const DeviceContext& ctx, const CommandPool& pool);

    [[nodiscard]] VkAccelerationStructureKHR tlas() const noexcept { return m_tlas.handle(); }
    [[nodiscard]] VkDeviceAddress tlasAddress() const noexcept { return m_tlasAddress; }
    [[nodiscard]] const Buffer& instanceBuffer() const noexcept { return m_instanceBuffer; }
    [[nodiscard]] const Buffer& materialBuffer() const noexcept { return m_materialBuffer; }
    [[nodiscard]] const Buffer& vertexBuffer() const noexcept { return m_vertexBuffer; }
    [[nodiscard]] const Buffer& indexBuffer() const noexcept { return m_indexBuffer; }
    [[nodiscard]] const Buffer& lightBuffer() const noexcept { return m_lightBuffer; }
    [[nodiscard]] const Buffer& emissiveTriangleBuffer() const noexcept { return m_emissiveTriangleBuffer; }
    [[nodiscard]] const Buffer& emissiveCdfBuffer() const noexcept { return m_emissiveCdfBuffer; }
    [[nodiscard]] uint32_t instanceCount() const noexcept { return static_cast<uint32_t>(m_geometries.size()); }
    [[nodiscard]] uint32_t lightCount() const noexcept { return static_cast<uint32_t>(m_lights.size()); }
    [[nodiscard]] uint32_t emissiveTriangleCount() const noexcept { return m_emissiveTriangleCount; }

  private:
    VkResult buildSceneBuffers(const DeviceContext& ctx, const CommandPool& pool);
    VkResult buildTlas(const DeviceContext& ctx, const CommandPool& pool);

    std::vector<GpuInstance> m_instances;
    Buffer m_instanceBuffer{};
    Buffer m_materialBuffer{};
    Buffer m_vertexBuffer{};
    Buffer m_indexBuffer{};
    Buffer m_lightBuffer{};
    Buffer m_emissiveTriangleBuffer{};
    Buffer m_emissiveCdfBuffer{};
    uint32_t m_emissiveTriangleCount = 0;
    AccelerationStructure m_tlas{};
    VkDeviceAddress m_tlasAddress{};
};
