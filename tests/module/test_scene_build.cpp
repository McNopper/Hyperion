// Module tests: Scene acceleration structure build (BLAS + TLAS).
//
// These tests verify that Scene::build() correctly builds BLASes and a TLAS
// for different geometry combinations and returns a non-null TLAS handle.
//
// Regression target: VK_ERROR_DEVICE_LOST from scene.build() in Debug without
// the validation layer, caused by invalid Vulkan usage that the layer masked.

#include <limits>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "fixtures/VulkanTestFixture.hpp"
#include "hyperion/scene/Material.hpp"
#include "hyperion/scene/ProceduralGeometry.hpp"
#include "hyperion/scene/Scene.hpp"

// Mesh + sphere: the same geometry combination used in the integration render test.
TEST_F(RtFixture, Scene_BuildWithMeshAndSphere) {
    Scene scene;
    const uint32_t matDiffuse = scene.addMaterial(Material::diffuse(glm::vec3(0.8F), 1.0F));
    const uint32_t matMetal   = scene.addMaterial(Material::metal(glm::vec3(0.9F, 0.3F, 0.2F), 0.15F));

    MeshData box = ProceduralGeometry::makeBox(glm::vec3(2.0F, 0.1F, 2.0F), glm::mat4(1.0F));
    const uint32_t meshInst =
        scene.addMesh(deviceCtx(), commandPool(), std::move(box), matDiffuse, "test.box");
    ASSERT_NE(meshInst, std::numeric_limits<uint32_t>::max()) << "Failed to upload floor mesh";

    const uint32_t sphereInst =
        scene.addSphere(deviceCtx(), commandPool(), glm::vec3(0.0F, 0.5F, 0.0F), 0.5F, matMetal);
    ASSERT_NE(sphereInst, std::numeric_limits<uint32_t>::max()) << "Failed to add sphere";

    const VkResult result = scene.build(deviceCtx(), commandPool());
    ASSERT_EQ(result, VK_SUCCESS) << "scene.build() failed: VkResult=" << static_cast<int>(result);

    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE) << "TLAS handle must be valid after build";
    EXPECT_NE(scene.tlasAddress(), VkDeviceAddress{0}) << "TLAS device address must be non-zero";
    EXPECT_EQ(scene.instanceCount(), 2U);
}

// Mesh-only scene: no analytic sphere, single BLAS type.
TEST_F(RtFixture, Scene_BuildWithMeshOnly) {
    Scene scene;
    const uint32_t mat = scene.addMaterial(Material::diffuse(glm::vec3(0.5F), 1.0F));

    MeshData tri = ProceduralGeometry::makeBox(glm::vec3(1.0F), glm::mat4(1.0F));
    const uint32_t inst = scene.addMesh(deviceCtx(), commandPool(), std::move(tri), mat, "test.solo");
    ASSERT_NE(inst, std::numeric_limits<uint32_t>::max());

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    EXPECT_EQ(scene.instanceCount(), 1U);
}

// Sphere-only scene: only AABB-based BLASes, no triangle geometry.
TEST_F(RtFixture, Scene_BuildWithSphereOnly) {
    Scene scene;
    const uint32_t mat = scene.addMaterial(Material::metal(glm::vec3(0.9F, 0.1F, 0.1F), 0.2F));

    const uint32_t inst =
        scene.addSphere(deviceCtx(), commandPool(), glm::vec3(0.0F), 1.0F, mat);
    ASSERT_NE(inst, std::numeric_limits<uint32_t>::max());

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    EXPECT_EQ(scene.instanceCount(), 1U);
}

// Multiple meshes: ensures that TLAS instance buffer correctly handles N>1 entries.
TEST_F(RtFixture, Scene_BuildWithMultipleMeshes) {
    Scene scene;
    const uint32_t mat = scene.addMaterial(Material::diffuse(glm::vec3(0.7F), 1.0F));

    for (int i = 0; i < 4; ++i) {
        const glm::mat4 transform =
            glm::translate(glm::mat4(1.0F), glm::vec3(static_cast<float>(i) * 2.0F, 0.0F, 0.0F));
        MeshData box = ProceduralGeometry::makeBox(glm::vec3(0.8F), transform);
        const uint32_t inst =
            scene.addMesh(deviceCtx(), commandPool(), std::move(box), mat, "test.multi");
        ASSERT_NE(inst, std::numeric_limits<uint32_t>::max()) << "mesh " << i;
    }

    ASSERT_EQ(scene.build(deviceCtx(), commandPool()), VK_SUCCESS);
    EXPECT_NE(scene.tlas(), VK_NULL_HANDLE);
    EXPECT_EQ(scene.instanceCount(), 4U);
}
