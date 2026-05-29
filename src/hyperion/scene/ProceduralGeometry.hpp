#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "hyperion/scene/Mesh.hpp"

namespace ProceduralGeometry {
[[nodiscard]] MeshData makeBox(glm::vec3 halfExtent, glm::mat4 transform = glm::mat4(1.0f));

struct SphereAabb {
    glm::vec3 min{};
    glm::vec3 max{};
};

[[nodiscard]] SphereAabb makeSphereAabb(glm::vec3 center, float radius) noexcept;
} // namespace ProceduralGeometry
