#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "hyperion/scene/ProceduralGeometry.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cstdint>

namespace ProceduralGeometry {
MeshData makeBox(glm::vec3 halfExtent, glm::mat4 transform) {
    struct Face {
        glm::vec3 normal;
        std::array<glm::vec3, 4> positions;
    };

    const glm::vec3 hx = {halfExtent.x, 0.0f, 0.0f};
    const glm::vec3 hy = {0.0f, halfExtent.y, 0.0f};
    const glm::vec3 hz = {0.0f, 0.0f, halfExtent.z};

    const std::array<Face, 6> faces{{
        {{1.0f, 0.0f, 0.0f}, {hx - hy - hz, hx + hy - hz, hx + hy + hz, hx - hy + hz}},
        {{-1.0f, 0.0f, 0.0f}, {-hx - hy + hz, -hx + hy + hz, -hx + hy - hz, -hx - hy - hz}},
        {{0.0f, 1.0f, 0.0f}, {-hx + hy - hz, -hx + hy + hz, hx + hy + hz, hx + hy - hz}},
        {{0.0f, -1.0f, 0.0f}, {-hx - hy + hz, -hx - hy - hz, hx - hy - hz, hx - hy + hz}},
        {{0.0f, 0.0f, 1.0f}, {-hx - hy + hz, hx - hy + hz, hx + hy + hz, -hx + hy + hz}},
        {{0.0f, 0.0f, -1.0f}, {hx - hy - hz, -hx - hy - hz, -hx + hy - hz, hx + hy - hz}},
    }};

    const glm::mat3 normalMatrix = glm::inverseTranspose(glm::mat3(transform));
    const std::array<glm::vec2, 4> uvs{{
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
    }};

    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    for (const Face& face : faces) {
        const uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
        const glm::vec3 transformedNormal = glm::normalize(normalMatrix * face.normal);

        for (size_t i = 0; i < face.positions.size(); ++i) {
            const glm::vec4 transformedPosition = transform * glm::vec4(face.positions[i], 1.0f);
            mesh.vertices.push_back(GpuVertex{
                .position = glm::vec3(transformedPosition),
                .tangentX = 0.0f,
                .normal = transformedNormal,
                .tangentY = 0.0f,
                .uv = uvs[i],
                .tangentZ = 0.0f,
                .bitangentSign = 1.0f,
            });
        }

        mesh.indices.insert(mesh.indices.end(),
                            {
                                baseVertex + 0U,
                                baseVertex + 1U,
                                baseVertex + 2U,
                                baseVertex + 0U,
                                baseVertex + 2U,
                                baseVertex + 3U,
                            });
    }

    return mesh;
}

SphereAabb makeSphereAabb(glm::vec3 center, float radius) noexcept {
    const glm::vec3 extent(std::max(radius, 0.0f));
    return SphereAabb{
        .min = center - extent,
        .max = center + extent,
    };
}
} // namespace ProceduralGeometry
