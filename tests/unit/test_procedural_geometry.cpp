#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

#include "harmonia/scene/ProceduralGeometry.hpp"
#include "harmonia/utils/Math.hpp"

namespace {
constexpr float kEpsilon = 1.0e-5F;

[[nodiscard]] bool matchesAnyDirection(glm::vec3 value, const std::array<glm::vec3, 6>& directions) noexcept {
    return std::any_of(directions.begin(), directions.end(), [&](const glm::vec3 direction) {
        return glm::length(value - direction) <= 1.0e-4F;
    });
}
} // namespace

TEST(ProceduralGeometry, MakeBoxIdentityProducesValidBoxMesh) {
    const MeshData mesh = ProceduralGeometry::makeBox(glm::vec3(1.0F), glm::mat4(1.0F));

    EXPECT_TRUE(mesh.vertices.size() == 24U || mesh.vertices.size() == 8U);
    EXPECT_EQ(mesh.indices.size(), 36U);

    const std::array axisDirections{
        glm::vec3(1.0F, 0.0F, 0.0F),
        glm::vec3(-1.0F, 0.0F, 0.0F),
        glm::vec3(0.0F, 1.0F, 0.0F),
        glm::vec3(0.0F, -1.0F, 0.0F),
        glm::vec3(0.0F, 0.0F, 1.0F),
        glm::vec3(0.0F, 0.0F, -1.0F),
    };

    for (const GpuVertex& vertex : mesh.vertices) {
        EXPECT_NEAR(glm::length(vertex.normal), 1.0F, kEpsilon);
        EXPECT_TRUE(matchesAnyDirection(vertex.normal, axisDirections));
        EXPECT_GE(vertex.position.x, -1.0F - kEpsilon);
        EXPECT_GE(vertex.position.y, -1.0F - kEpsilon);
        EXPECT_GE(vertex.position.z, -1.0F - kEpsilon);
        EXPECT_LE(vertex.position.x, 1.0F + kEpsilon);
        EXPECT_LE(vertex.position.y, 1.0F + kEpsilon);
        EXPECT_LE(vertex.position.z, 1.0F + kEpsilon);
        EXPECT_GE(vertex.uv.x, 0.0F);
        EXPECT_GE(vertex.uv.y, 0.0F);
        EXPECT_LE(vertex.uv.x, 1.0F);
        EXPECT_LE(vertex.uv.y, 1.0F);
    }
}

TEST(ProceduralGeometry, MakeBoxRotationTransformsNormals) {
    const glm::mat4 rotation = Math::makeRotationY(Math::kPi * 0.5F);
    const MeshData mesh = ProceduralGeometry::makeBox(glm::vec3(1.0F), rotation);
    const glm::mat3 normalTransform(rotation);

    const std::array expectedDirections{
        glm::normalize(normalTransform * glm::vec3(1.0F, 0.0F, 0.0F)),
        glm::normalize(normalTransform * glm::vec3(-1.0F, 0.0F, 0.0F)),
        glm::normalize(normalTransform * glm::vec3(0.0F, 1.0F, 0.0F)),
        glm::normalize(normalTransform * glm::vec3(0.0F, -1.0F, 0.0F)),
        glm::normalize(normalTransform * glm::vec3(0.0F, 0.0F, 1.0F)),
        glm::normalize(normalTransform * glm::vec3(0.0F, 0.0F, -1.0F)),
    };

    for (const GpuVertex& vertex : mesh.vertices) {
        EXPECT_NEAR(glm::length(vertex.normal), 1.0F, kEpsilon);
        EXPECT_TRUE(matchesAnyDirection(vertex.normal, expectedDirections));
    }
}

TEST(ProceduralGeometry, MakeSphereAabbAtOriginMatchesUnitSphere) {
    const auto aabb = ProceduralGeometry::makeSphereAabb(glm::vec3(0.0F), 1.0F);
    EXPECT_NEAR(aabb.min.x, -1.0F, kEpsilon);
    EXPECT_NEAR(aabb.min.y, -1.0F, kEpsilon);
    EXPECT_NEAR(aabb.min.z, -1.0F, kEpsilon);
    EXPECT_NEAR(aabb.max.x, 1.0F, kEpsilon);
    EXPECT_NEAR(aabb.max.y, 1.0F, kEpsilon);
    EXPECT_NEAR(aabb.max.z, 1.0F, kEpsilon);
}

TEST(ProceduralGeometry, MakeSphereAabbWithOffsetMatchesExpectedBounds) {
    const auto aabb = ProceduralGeometry::makeSphereAabb(glm::vec3(1.0F, 2.0F, 3.0F), 0.5F);
    EXPECT_NEAR(aabb.min.x, 0.5F, kEpsilon);
    EXPECT_NEAR(aabb.min.y, 1.5F, kEpsilon);
    EXPECT_NEAR(aabb.min.z, 2.5F, kEpsilon);
    EXPECT_NEAR(aabb.max.x, 1.5F, kEpsilon);
    EXPECT_NEAR(aabb.max.y, 2.5F, kEpsilon);
    EXPECT_NEAR(aabb.max.z, 3.5F, kEpsilon);
}
