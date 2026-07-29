// Module tests: Scene acceleration structure build (BLAS + TLAS).
//
// These tests verify that Scene::build() correctly builds BLASes and a TLAS
// under the mesh/instance model (unique meshes registered once; instances placed
// with a transform) and returns a non-null TLAS handle.
//
// Scene::build() must be valid Vulkan usage even when the validation layer is absent —
// the layer can mask invalid usage that otherwise surfaces as VK_ERROR_DEVICE_LOST in Debug.

#include <gtest/gtest.h>
#include <limits>
#include <slang-math/slang-math.hpp>

#include "fixtures/VulkanTestFixture.hpp"
#include "harmonia/scene/Material.hpp"
#include "harmonia/scene/ProceduralGeometry.hpp"
#include "hyperion/scene/Scene.hpp"

// Mesh + sphere: the same geometry combination used in the integration render test.
TEST_F(RtFixture, Scene_BuildWithMeshAndSphere) {
    Scene scene;
    const uint32_t matDiffuse = scene.addMaterial(Material::diffuse(sm::float3(0.8F), 1.0F));
    const uint32_t matMetal = scene.addMaterial(Material::metal(sm::float3(0.9F, 0.3F, 0.2F), 0.15F));

    MeshData box = ProceduralGeometry::makeBox(sm::float3(2.0F, 0.1F, 2.0F)); // object space
    const uint32_t boxMesh = scene.addMesh(deviceCtx(), commandPool(), std::move(box), "test.box");
    ASSERT_NE(boxMesh, std::numeric_limits<uint32_t>::max()) << "Failed to upload floor mesh";

    const uint32_t sphereMesh = scene.addSphereMesh(deviceCtx(), commandPool(), 0.5F, "test.sphere");
    ASSERT_NE(sphereMesh, std::numeric_limits<uint32_t>::max()) << "Failed to add sphere mesh";

    ASSERT_NE(scene.addInstance(boxMesh, Xform{.translation = {0.0F, 0.0F, 0.0F}}, matDiffuse),
              std::numeric_limits<uint32_t>::max());
    ASSERT_NE(scene.addInstance(sphereMesh, Xform{.translation = {0.0F, 0.5F, 0.0F}}, matMetal),
              std::numeric_limits<uint32_t>::max());

    const VkResult result = scene.build(deviceCtx(), commandPool());
    ASSERT_EQ(result, VK_SUCCESS) << "scene.build() failed: VkResult=" << static_cast<int>(result);

    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE) << "TLAS handle must be valid after build";
    EXPECT_NE(scene.tlasAddress(), VkDeviceAddress{0}) << "TLAS device address must be non-zero";
    EXPECT_EQ(scene.instanceCount(), 2U);
}

// Mesh-only scene: no analytic sphere, single BLAS type.
TEST_F(RtFixture, Scene_BuildWithMeshOnly) {
    Scene scene;
    const uint32_t mat = scene.addMaterial(Material::diffuse(sm::float3(0.5F), 1.0F));

    MeshData tri = ProceduralGeometry::makeBox(sm::float3(1.0F));
    const uint32_t mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(tri), "test.solo");
    ASSERT_NE(mesh, std::numeric_limits<uint32_t>::max());
    ASSERT_NE(scene.addInstance(mesh, Xform{}, mat), std::numeric_limits<uint32_t>::max());

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    EXPECT_EQ(scene.instanceCount(), 1U);
}

// Sphere-only scene: only AABB-based BLASes, no triangle geometry.
TEST_F(RtFixture, Scene_BuildWithSphereOnly) {
    Scene scene;
    const uint32_t mat = scene.addMaterial(Material::metal(sm::float3(0.9F, 0.1F, 0.1F), 0.2F));

    const uint32_t mesh = scene.addSphereMesh(deviceCtx(), commandPool(), 1.0F, "test.soloSphere");
    ASSERT_NE(mesh, std::numeric_limits<uint32_t>::max());
    ASSERT_NE(scene.addInstance(mesh, Xform{}, mat), std::numeric_limits<uint32_t>::max());

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    EXPECT_EQ(scene.instanceCount(), 1U);
}

// One mesh instanced N times: the core instancing case — a single BLAS shared by
// multiple TLAS instances placed at different transforms.
TEST_F(RtFixture, Scene_BuildWithMultipleInstancesOfOneMesh) {
    Scene scene;
    const uint32_t mat = scene.addMaterial(Material::diffuse(sm::float3(0.7F), 1.0F));

    MeshData box = ProceduralGeometry::makeBox(sm::float3(0.8F)); // one unique mesh
    const uint32_t mesh = scene.addMesh(deviceCtx(), commandPool(), std::move(box), "test.shared");
    ASSERT_NE(mesh, std::numeric_limits<uint32_t>::max());

    for (std::size_t i = 0; i < 4; ++i) {
        const Xform xform{.translation = sm::float3(static_cast<float>(i) * 2.0F, 0.0F, 0.0F)};
        ASSERT_NE(scene.addInstance(mesh, xform, mat), std::numeric_limits<uint32_t>::max()) << "instance " << i;
    }

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    // 4 instances of one mesh → 4 TLAS entries referencing one shared BLAS.
    EXPECT_EQ(scene.instanceCount(), 4U);
}
