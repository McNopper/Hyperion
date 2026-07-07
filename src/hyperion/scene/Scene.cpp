#include "hyperion/scene/Scene.hpp"

#include <slang-math/slang-math.hpp>
#include <volk/volk.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vma/vk_mem_alloc.h>

#include "harmonia/GpuTypes.hpp"
#include "harmonia/core/Buffer.hpp"
#include "harmonia/core/Logger.hpp"
#include "harmonia/scene/EmissiveBuilder.hpp"
#include "harmonia/scene/Geometry.hpp"

uint32_t Scene::addMesh(const DeviceContext& ctx,
                        const CommandPool& pool,
                        MeshData&& data,
                        uint32_t materialIdx,
                        std::string_view name) {
    const uint32_t instanceIndex = static_cast<uint32_t>(m_geometries.size());
    const std::string debugName =
        name.empty() ? std::string{"mesh."} + std::to_string(instanceIndex) : std::string{name};

    auto mesh = TriangleMesh::create(ctx, pool, std::move(data), materialIdx, debugName);
    if (!mesh) {
        return std::numeric_limits<uint32_t>::max();
    }

    m_geometries.push_back(std::move(*mesh));
    m_instances.push_back(GpuInstance{
        .meshIndex = instanceIndex,
        .materialIndex = materialIdx,
        .vertexOffset = 0,
        .indexOffset = 0,
        .geometryKind = 0,
        .sphereRadius = 0.0f,
        ._pad = {0, 0},
    });
    return instanceIndex;
}

uint32_t Scene::addSphere(const DeviceContext& ctx,
                          const CommandPool& pool,
                          sm::float3 center,
                          float radius,
                          uint32_t materialIdx) {
    const uint32_t instanceIndex = static_cast<uint32_t>(m_geometries.size());
    const std::string debugName = std::string{"sphere."} + std::to_string(instanceIndex);

    auto sphere = Sphere::create(ctx, pool, center, radius, materialIdx, debugName);
    if (!sphere) {
        return std::numeric_limits<uint32_t>::max();
    }

    m_geometries.push_back(std::move(*sphere));
    m_instances.push_back(GpuInstance{
        .meshIndex = instanceIndex,
        .materialIndex = materialIdx,
        .vertexOffset = 0,
        .indexOffset = 0,
        .geometryKind = 1,
        .sphereRadius = radius,
        ._pad = {0, 0},
    });
    return instanceIndex;
}

VkResult Scene::build(const DeviceContext& ctx, const CommandPool& pool) {
    if (m_geometries.empty()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    assert(m_geometries.size() == m_instances.size());
    if (const VkResult result = buildSceneBuffers(ctx, pool); result != VK_SUCCESS) {
        return result;
    }
    for (auto& geo : m_geometries) {
        if (const VkResult result = geo->buildBlas(ctx, pool); result != VK_SUCCESS) {
            return result;
        }
    }
    return buildTlas(ctx, pool);
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

    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(std::max<size_t>(m_geometries.size(), 1));

    for (size_t i = 0; i < m_geometries.size(); ++i) {
        GpuInstance& instance = m_instances[i];

        if (const auto* mesh = dynamic_cast<const TriangleMesh*>(m_geometries[i].get())) {
            instance.vertexOffset = static_cast<uint32_t>(vertices.size());
            instance.indexOffset = static_cast<uint32_t>(indices.size());
            vertices.insert(vertices.end(), mesh->data().vertices.begin(), mesh->data().vertices.end());
            for (uint32_t index : mesh->data().indices) {
                indices.push_back(index + instance.vertexOffset);
            }
        } else if (const auto* sphere = dynamic_cast<const Sphere*>(m_geometries[i].get())) {
            instance.vertexOffset = static_cast<uint32_t>(vertices.size());
            instance.indexOffset = 0;
            // Store sphere centre in the vertex position slot for shader lookup.
            vertices.push_back(GpuVertex{
                .position = sphere->center(),
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

    auto instanceBuf =
        Buffer::upload(ctx,
                       pool,
                       std::as_bytes(std::span<const GpuInstance>(m_instances.data(), m_instances.size())),
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
    std::vector<harmonia::EmissiveInstanceInfo> emissiveInstances;
    emissiveInstances.reserve(m_instances.size());
    for (const GpuInstance& inst : m_instances) {
        emissiveInstances.push_back({inst.geometryKind, inst.materialIndex});
    }
    harmonia::EmissiveData emissiveData =
        harmonia::buildEmissiveData(m_geometries, emissiveInstances, m_materials, gpuMaterials);

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

    // Power-proportional selection CDF: cdf[i] = (Σ_{j≤i} power_j) / totalPower, so cdf[N-1] == 1.
    // Falls back to a uniform CDF when all emitters have zero power (degenerate/black emitters).
    const std::vector<float>& emissivePower = emissiveData.power;
    std::vector<float> emissiveCdf;
    emissiveCdf.reserve(emissivePower.size());
    double totalPower = 0.0;
    for (const float p : emissivePower) {
        totalPower += static_cast<double>(p);
    }
    if (totalPower > 0.0) {
        double running = 0.0;
        for (const float p : emissivePower) {
            running += static_cast<double>(p);
            emissiveCdf.push_back(static_cast<float>(running / totalPower));
        }
        if (!emissiveCdf.empty()) {
            emissiveCdf.back() = 1.0F; // guard against rounding leaving cdf[N-1] < 1
        }
    } else {
        const auto count = static_cast<uint32_t>(emissivePower.size());
        for (uint32_t i = 0; i < count; ++i) {
            emissiveCdf.push_back(static_cast<float>(i + 1) / static_cast<float>(count));
        }
    }
    if (emissiveCdf.empty()) {
        emissiveCdf.push_back(1.0F); // sentinel — keeps the binding valid when no emitters
    }
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
    std::vector<VkAccelerationStructureInstanceKHR> instances(m_geometries.size());
    for (size_t i = 0; i < m_geometries.size(); ++i) {
        instances[i] = m_geometries[i]->makeInstance(static_cast<uint32_t>(i));
    }

    auto instanceUpload = Buffer::upload(
        ctx,
        pool,
        std::as_bytes(std::span<const VkAccelerationStructureInstanceKHR>(instances.data(), instances.size())),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        "scene.tlas.instances");
    if (!instanceUpload) {
        return instanceUpload.error();
    }

    const VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .pNext = nullptr,
        .arrayOfPointers = VK_FALSE,
        .data = VkDeviceOrHostAddressConstKHR{instanceUpload->deviceAddress()},
    };
    const VkAccelerationStructureGeometryKHR geometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext = nullptr,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry = VkAccelerationStructureGeometryDataKHR{.instances = instancesData},
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    const uint32_t primitiveCount = static_cast<uint32_t>(instances.size());
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .pNext = nullptr,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount = 1,
        .pGeometries = &geometry,
        .ppGeometries = nullptr,
        .scratchData = VkDeviceOrHostAddressKHR{},
    };
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(
        ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

    auto tlasAS = AccelerationStructure::create(
        ctx, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo.accelerationStructureSize, "scene.tlas");
    if (!tlasAS) {
        return tlasAS.error();
    }

    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
    asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props{};
    props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props.pNext = &asProps;
    vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &props);

    auto scratch = Buffer::create(
        ctx,
        std::max<VkDeviceSize>(sizeInfo.buildScratchSize + asProps.minAccelerationStructureScratchOffsetAlignment, 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        "scene.tlasScratch");
    if (!scratch) {
        return scratch.error();
    }

    buildInfo.dstAccelerationStructure = tlasAS->handle();
    buildInfo.scratchData.deviceAddress =
        bufferAlignUp(scratch->deviceAddress(), asProps.minAccelerationStructureScratchOffsetAlignment);
    const VkAccelerationStructureBuildRangeInfoKHR rangeInfo{
        .primitiveCount = primitiveCount,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };
    const VkAccelerationStructureBuildRangeInfoKHR* rangePtr = &rangeInfo;

    auto cmd = pool.beginOneShot();
    if (!cmd) {
        return cmd.error();
    }
    vkCmdBuildAccelerationStructuresKHR(*cmd, 1, &buildInfo, &rangePtr);
    if (const VkResult result = pool.endOneShot(*cmd); result != VK_SUCCESS) {
        return result;
    }

    m_tlas = std::move(*tlasAS);
    m_tlasAddress = m_tlas.deviceAddress();
    return VK_SUCCESS;
}
