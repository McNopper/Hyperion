#include "hyperion/scene/Scene.hpp"

#include <volk/volk.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <slang-math/slang-math.hpp>
#include <span>
#include <string_view>
#include <utility>
#include <vma/vk_mem_alloc.h>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/renderer/TlasBuilder.hpp"
#include "harmonia/scene/EmissiveBuilder.hpp"
#include "harmonia/scene/Geometry.hpp"

std::uint32_t Scene::addSphereMesh(const harmonia::DeviceContext& ctx,
                                   const harmonia::CommandPool& pool,
                                   float radius,
                                   std::string_view name) {
    const std::uint32_t meshIndex = static_cast<std::uint32_t>(m_meshes.size());
    const std::string debugName = name.empty() ? std::string{"sphere."} + std::to_string(meshIndex) : std::string{name};

    auto sphere = harmonia::Sphere::create(ctx, pool, radius, debugName);
    if (!sphere) {
        return std::numeric_limits<std::uint32_t>::max();
    }
    m_meshes.push_back(std::move(*sphere));
    return meshIndex;
}

VkResult Scene::buildSceneBuffers(const harmonia::DeviceContext& ctx, const harmonia::CommandPool& pool) {
    std::vector<harmonia::GpuMaterial> gpuMaterials;
    gpuMaterials.reserve(std::max<std::size_t>(m_materials.size(), 1));
    for (const harmonia::Material& material : m_materials) {
        gpuMaterials.push_back(material.gpu());
    }
    if (gpuMaterials.empty()) {
        gpuMaterials.push_back(harmonia::GpuMaterial{});
    }

    // Lay out the global vertex/index buffers from the unique meshes (object space),
    // recording each mesh's range for the per-instance GpuInstance rows.
    std::vector<harmonia::GpuVertex> vertices;
    std::vector<std::uint32_t> indices;
    m_meshGpu.assign(m_meshes.size(), MeshGpu{});

    for (std::size_t mi = 0; mi < m_meshes.size(); ++mi) {
        if (const auto* mesh = dynamic_cast<const harmonia::TriangleMesh*>(m_meshes[mi].get())) {
            MeshGpu& gpu = m_meshGpu[mi];
            gpu.vertexOffset = static_cast<std::uint32_t>(vertices.size());
            gpu.indexOffset = static_cast<std::uint32_t>(indices.size());
            gpu.geometryKind = 0;
            vertices.insert(vertices.end(), mesh->data().vertices.begin(), mesh->data().vertices.end());
            for (std::uint32_t index : mesh->data().indices) {
                indices.push_back(index + gpu.vertexOffset);
            }
        } else if (const auto* sphere = dynamic_cast<const harmonia::Sphere*>(m_meshes[mi].get())) {
            MeshGpu& gpu = m_meshGpu[mi];
            gpu.vertexOffset = static_cast<std::uint32_t>(vertices.size());
            gpu.indexOffset = 0;
            gpu.geometryKind = 1;
            gpu.sphereRadius = sphere->radius();
            // Object-space sphere lives at the origin; the instance transform places it.
            vertices.push_back(harmonia::GpuVertex{
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
        vertices.push_back(harmonia::GpuVertex{});
    }
    if (indices.empty()) {
        indices.push_back(0);
    }

    buildGpuInstances();

    // harmonia::Light buffer — always upload at least one sentinel entry so the binding is valid.
    std::vector<harmonia::GpuLight> gpuLights;
    gpuLights.reserve(std::max<std::size_t>(m_lights.size(), 1));
    for (const auto& light : m_lights) {
        gpuLights.push_back(light->toGpu());
    }
    if (gpuLights.empty()) {
        gpuLights.push_back(harmonia::GpuLight{});
    }

    // Build emissive triangle buffer via the shared harmonia utility.
    harmonia::EmissiveData emissiveData = harmonia::buildEmissiveData(m_meshes, m_instances, m_materials, gpuMaterials);

    m_emissiveTriangleCount = static_cast<std::uint32_t>(emissiveData.triangles.size());
    harmonia::Logger::info("Scene: built {} emissive triangle(s) for NEE", m_emissiveTriangleCount);
    if (emissiveData.triangles.empty()) {
        emissiveData.triangles.push_back(harmonia::GpuEmissiveTriangle{}); // sentinel — keeps the binding valid
    }

    // Power-proportional selection CDF for emissive-triangle NEE (shared helper —
    // identical across renderers, so the NEE sampling CDF cannot drift).
    const std::vector<float> emissiveCdf = harmonia::buildEmissiveCdf(emissiveData.power);

    return uploadAllBuffers(ctx, pool, gpuMaterials, vertices, indices, gpuLights, emissiveData.triangles, emissiveCdf);
}

void Scene::buildGpuInstances() {
    // Build per-instance GPU rows from the instance list (mesh index + material).
    m_gpuInstances.clear();
    m_gpuInstances.reserve(m_instances.size());
    for (const harmonia::InstanceRecord& inst : m_instances) {
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
}

VkResult Scene::uploadAllBuffers(const harmonia::DeviceContext& ctx,
                                 const harmonia::CommandPool& pool,
                                 const std::vector<harmonia::GpuMaterial>& gpuMaterials,
                                 const std::vector<harmonia::GpuVertex>& vertices,
                                 const std::vector<std::uint32_t>& indices,
                                 const std::vector<harmonia::GpuLight>& gpuLights,
                                 const std::vector<harmonia::GpuEmissiveTriangle>& emissiveTriangles,
                                 const std::vector<float>& emissiveCdf) {
    constexpr VkBufferUsageFlags kStorageAddr =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    auto instanceBuf =
        uploadStorageBuffer(ctx,
                            pool,
                            std::as_bytes(std::span<const GpuInstance>(m_gpuInstances.data(), m_gpuInstances.size())),
                            "scene.instances",
                            kStorageAddr);
    if (!instanceBuf) {
        return instanceBuf.error();
    }

    auto materialBuf = uploadStorageBuffer(
        ctx,
        pool,
        std::as_bytes(std::span<const harmonia::GpuMaterial>(gpuMaterials.data(), gpuMaterials.size())),
        "scene.materials",
        kStorageAddr);
    if (!materialBuf) {
        return materialBuf.error();
    }

    auto vertexBuf =
        uploadStorageBuffer(ctx,
                            pool,
                            std::as_bytes(std::span<const harmonia::GpuVertex>(vertices.data(), vertices.size())),
                            "scene.vertices",
                            kStorageAddr);
    if (!vertexBuf) {
        return vertexBuf.error();
    }

    auto indexBuf = uploadStorageBuffer(ctx,
                                        pool,
                                        std::as_bytes(std::span<const std::uint32_t>(indices.data(), indices.size())),
                                        "scene.indices",
                                        kStorageAddr);
    if (!indexBuf) {
        return indexBuf.error();
    }

    m_instanceBuffer = std::move(*instanceBuf);
    m_materialBuffer = std::move(*materialBuf);
    m_vertexBuffer = std::move(*vertexBuf);
    m_indexBuffer = std::move(*indexBuf);

    auto lightBuf =
        uploadStorageBuffer(ctx,
                            pool,
                            std::as_bytes(std::span<const harmonia::GpuLight>(gpuLights.data(), gpuLights.size())),
                            "scene.lights",
                            kStorageAddr);
    if (!lightBuf) {
        return lightBuf.error();
    }
    m_lightBuffer = std::move(*lightBuf);

    auto emissiveBuf = uploadStorageBuffer(ctx,
                                           pool,
                                           std::as_bytes(std::span<const harmonia::GpuEmissiveTriangle>(
                                               emissiveTriangles.data(), emissiveTriangles.size())),
                                           "scene.emissiveTriangles",
                                           kStorageAddr);
    if (!emissiveBuf) {
        return emissiveBuf.error();
    }
    m_emissiveTriangleBuffer = std::move(*emissiveBuf);

    auto emissiveCdfBuf =
        uploadStorageBuffer(ctx,
                            pool,
                            std::as_bytes(std::span<const float>(emissiveCdf.data(), emissiveCdf.size())),
                            "scene.emissiveCdf",
                            kStorageAddr);
    if (!emissiveCdfBuf) {
        return emissiveCdfBuf.error();
    }
    m_emissiveCdfBuffer = std::move(*emissiveCdfBuf);

    return VK_SUCCESS;
}
