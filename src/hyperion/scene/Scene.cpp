#include "hyperion/scene/Scene.hpp"

#include <slang-math/slang-math.hpp>
#include <volk/volk.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vma/vk_mem_alloc.h>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/scene/EmissiveBuilder.hpp"
#include "harmonia/scene/Geometry.hpp"
#include "harmonia/renderer/TlasBuilder.hpp"

uint32_t Scene::addSphereMesh(const DeviceContext& ctx,
                              const CommandPool& pool,
                              float radius,
                              std::string_view name) {
    const uint32_t meshIndex = static_cast<uint32_t>(m_meshes.size());
    const std::string debugName =
        name.empty() ? std::string{"sphere."} + std::to_string(meshIndex) : std::string{name};

    auto sphere = Sphere::create(ctx, pool, radius, debugName);
    if (!sphere) {
        return std::numeric_limits<uint32_t>::max();
    }
    m_meshes.push_back(std::move(*sphere));
    return meshIndex;
}

VkResult Scene::buildSceneBuffers(const DeviceContext& ctx, const CommandPool& pool) {
    std::vector<GpuMaterial> gpuMaterials;
    gpuMaterials.reserve(std::max<size_t>(m_materials.size(), 1));
    for (const Material& material : m_materials) {
        gpuMaterials.push_back(material.gpu());
    }
    if (gpuMaterials.empty()) {
        gpuMaterials.push_back(GpuMaterial{});
    }

    // Lay out the global vertex/index buffers from the unique meshes (object space),
    // recording each mesh's range for the per-instance GpuInstance rows.
    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
    m_meshGpu.assign(m_meshes.size(), MeshGpu{});

    for (size_t mi = 0; mi < m_meshes.size(); ++mi) {
        if (const auto* mesh = dynamic_cast<const TriangleMesh*>(m_meshes[mi].get())) {
            MeshGpu& gpu = m_meshGpu[mi];
            gpu.vertexOffset = static_cast<uint32_t>(vertices.size());
            gpu.indexOffset = static_cast<uint32_t>(indices.size());
            gpu.geometryKind = 0;
            vertices.insert(vertices.end(), mesh->data().vertices.begin(), mesh->data().vertices.end());
            for (uint32_t index : mesh->data().indices) {
                indices.push_back(index + gpu.vertexOffset);
            }
        } else if (const auto* sphere = dynamic_cast<const Sphere*>(m_meshes[mi].get())) {
            MeshGpu& gpu = m_meshGpu[mi];
            gpu.vertexOffset = static_cast<uint32_t>(vertices.size());
            gpu.indexOffset = 0;
            gpu.geometryKind = 1;
            gpu.sphereRadius = sphere->radius();
            // Object-space sphere lives at the origin; the instance transform places it.
            vertices.push_back(GpuVertex{
                .position = sm::float3{0.0f, 0.0f, 0.0f},
                .tangentX = 0.0f,
                .normal = sm::float3{0.0f, 0.0f, 0.0f},
                .tangentY = 0.0f,
                .uv = sm::float2{0.0f, 0.0f},
                .tangentZ = 0.0f,
                .bitangentSign = 1.0f,
            });
        }
    }

    if (vertices.empty()) {
        vertices.push_back(GpuVertex{});
    }
    if (indices.empty()) {
        indices.push_back(0);
    }

    // Build per-instance GPU rows from the instance list (mesh index + material).
    m_gpuInstances.clear();
    m_gpuInstances.reserve(m_instances.size());
    for (const InstanceRecord& inst : m_instances) {
        const MeshGpu& gpu = m_meshGpu[inst.meshIndex];
        m_gpuInstances.push_back(GpuInstance{
            .meshIndex = inst.meshIndex,
            .materialIndex = inst.materialIndex,
            .vertexOffset = gpu.vertexOffset,
            .indexOffset = gpu.indexOffset,
            .geometryKind = gpu.geometryKind,
            .sphereRadius = gpu.sphereRadius,
            ._pad = {0, 0},
        });
    }

    auto instanceBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const GpuInstance>(m_gpuInstances.data(), m_gpuInstances.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.instances");
    if (!instanceBuf) {
        return instanceBuf.error();
    }

    auto materialBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const GpuMaterial>(gpuMaterials.data(), gpuMaterials.size())),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                       "scene.materials");
    if (!materialBuf) {
        return materialBuf.error();
    }

    auto vertexBuf = Buffer::upload(ctx,
                                    pool,
                                    std::as_bytes(std::span<const GpuVertex>(vertices.data(), vertices.size())),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    "scene.vertices");
    if (!vertexBuf) {
        return vertexBuf.error();
    }

    auto indexBuf = Buffer::upload(ctx,
                                   pool,
                                   std::as_bytes(std::span<const uint32_t>(indices.data(), indices.size())),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   "scene.indices");
    if (!indexBuf) {
        return indexBuf.error();
    }

    m_instanceBuffer = std::move(*instanceBuf);
    m_materialBuffer = std::move(*materialBuf);
    m_vertexBuffer = std::move(*vertexBuf);
    m_indexBuffer = std::move(*indexBuf);

    // Light buffer — always upload at least one sentinel entry so the binding is valid.
    std::vector<GpuLight> gpuLights;
    gpuLights.reserve(std::max<size_t>(m_lights.size(), 1));
    for (const auto& light : m_lights) {
        gpuLights.push_back(light->toGpu());
    }
    if (gpuLights.empty()) {
        gpuLights.push_back(GpuLight{});
    }
    auto lightBuf = Buffer::upload(ctx,
                                   pool,
                                   std::as_bytes(std::span<const GpuLight>(gpuLights.data(), gpuLights.size())),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                   "scene.lights");
    if (!lightBuf) {
        return lightBuf.error();
    }
    m_lightBuffer = std::move(*lightBuf);

    // Build emissive triangle buffer via the shared harmonia utility.
    harmonia::EmissiveData emissiveData = harmonia::buildEmissiveData(m_meshes, m_instances, m_materials, gpuMaterials);

    m_emissiveTriangleCount = static_cast<uint32_t>(emissiveData.triangles.size());
    Logger::info("Scene: built {} emissive triangle(s) for NEE", m_emissiveTriangleCount);
    if (emissiveData.triangles.empty()) {
        emissiveData.triangles.push_back(GpuEmissiveTriangle{}); // sentinel — keeps the binding valid
    }
    auto emissiveBuf = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const GpuEmissiveTriangle>(emissiveData.triangles.data(), emissiveData.triangles.size())),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        "scene.emissiveTriangles");
    if (!emissiveBuf) {
        return emissiveBuf.error();
    }
    m_emissiveTriangleBuffer = std::move(*emissiveBuf);

    // Power-proportional selection CDF for emissive-triangle NEE (shared helper —
    // identical across renderers, so the NEE sampling CDF cannot drift).
    const std::vector<float> emissiveCdf = harmonia::buildEmissiveCdf(emissiveData.power);
    auto emissiveCdfBuf = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const float>(emissiveCdf.data(), emissiveCdf.size())),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        "scene.emissiveCdf");
    if (!emissiveCdfBuf) {
        return emissiveCdfBuf.error();
    }
    m_emissiveCdfBuffer = std::move(*emissiveCdfBuf);

    return VK_SUCCESS;
}

VkResult Scene::buildTlas(const DeviceContext& ctx, const CommandPool& pool) {
    // One TLAS instance per InstanceRecord, each referencing its mesh's shared BLAS
    // and carrying that instance's object→world transform. Hyperion uses the default
    // instance mask (makeInstance); Theia stamps emissive/transparent/opaque masks.
    std::vector<VkAccelerationStructureInstanceKHR> instances(m_instances.size());
    for (size_t i = 0; i < m_instances.size(); ++i) {
        const InstanceRecord& inst = m_instances[i];
        instances[i] = m_meshes[inst.meshIndex]->makeInstance(static_cast<uint32_t>(i), inst.xform);
    }
    return harmonia::buildTlas(ctx, pool, instances, m_tlas, m_tlasAddress);
}
