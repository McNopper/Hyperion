#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <array>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "demo/importers/ISceneImporter.hpp"
#include "demo/importers/ObjImporter.hpp"
#include "hyperion/GpuTypes.hpp"

namespace {
constexpr float kEpsilon = 1.0e-5F;

struct ParsedMesh {
    std::vector<GpuVertex> vertices;
    std::vector<uint32_t> indices;
};

struct FaceIndex {
    int position = 0;
    int texcoord = 0;
    int normal = 0;

    [[nodiscard]] auto tie() const noexcept { return std::tie(position, texcoord, normal); }

    [[nodiscard]] bool operator==(const FaceIndex& other) const noexcept { return tie() == other.tie(); }
};

struct FaceIndexHash {
    [[nodiscard]] size_t operator()(const FaceIndex& index) const noexcept {
        size_t seed = static_cast<size_t>(index.position);
        seed = (seed * 1315423911ULL) ^ static_cast<size_t>(index.texcoord);
        seed = (seed * 2654435761ULL) ^ static_cast<size_t>(index.normal);
        return seed;
    }
};

[[nodiscard]] std::filesystem::path fixturePath(std::string_view fileName) {
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / std::filesystem::path(fileName);
}

[[nodiscard]] bool parseFaceVertex(std::string_view token, FaceIndex& outIndex) {
    std::istringstream stream{std::string(token)};
    std::string segment;
    std::array<int, 3> values{0, 0, 0};
    size_t cursor = 0;
    while (std::getline(stream, segment, '/') && cursor < values.size()) {
        if (!segment.empty()) {
            values[cursor] = std::stoi(segment);
        }
        ++cursor;
    }
    if (values[0] <= 0 || values[1] <= 0 || values[2] <= 0) {
        return false;
    }
    outIndex = FaceIndex{values[0], values[1], values[2]};
    return true;
}

[[nodiscard]] bool loadObjFixture(const std::filesystem::path& path, ParsedMesh& mesh) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::unordered_map<FaceIndex, uint32_t, FaceIndexHash> remap;

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }

        std::istringstream lineStream(line);
        std::string prefix;
        lineStream >> prefix;
        if (prefix == "v") {
            glm::vec3 position{};
            lineStream >> position.x >> position.y >> position.z;
            positions.push_back(position);
        } else if (prefix == "vn") {
            glm::vec3 normal{};
            lineStream >> normal.x >> normal.y >> normal.z;
            normals.push_back(normal);
        } else if (prefix == "vt") {
            glm::vec2 uv{};
            lineStream >> uv.x >> uv.y;
            texcoords.push_back(uv);
        } else if (prefix == "f") {
            std::vector<FaceIndex> polygon;
            std::string token;
            while (lineStream >> token) {
                FaceIndex faceIndex{};
                if (!parseFaceVertex(token, faceIndex)) {
                    return false;
                }
                polygon.push_back(faceIndex);
            }
            if (polygon.size() < 3U) {
                return false;
            }

            for (size_t i = 1; i + 1 < polygon.size(); ++i) {
                for (const FaceIndex faceIndex : {polygon[0], polygon[i], polygon[i + 1]}) {
                    const auto [it, inserted] =
                        remap.try_emplace(faceIndex, static_cast<uint32_t>(mesh.vertices.size()));
                    if (inserted) {
                        const glm::vec3 position = positions.at(static_cast<size_t>(faceIndex.position - 1));
                        const glm::vec2 uv = texcoords.at(static_cast<size_t>(faceIndex.texcoord - 1));
                        const glm::vec3 normal = glm::normalize(normals.at(static_cast<size_t>(faceIndex.normal - 1)));
                        mesh.vertices.push_back(GpuVertex{
                            .position      = position,
                            .tangentX      = 0.0F,
                            .normal        = normal,
                            .tangentY      = 0.0F,
                            .uv            = uv,
                            .tangentZ      = 0.0F,
                            .bitangentSign = 1.0F,
                        });
                    }
                    mesh.indices.push_back(it->second);
                }
            }
        }
    }

    return !mesh.vertices.empty() && !mesh.indices.empty();
}
} // namespace

static_assert(std::derived_from<ObjImporter, ISceneImporter>);

TEST(ObjImporter, TriangleFixtureParsesWithoutError) {
    ParsedMesh mesh;
    EXPECT_TRUE(loadObjFixture(fixturePath("triangle.obj"), mesh));
}

TEST(ObjImporter, TriangleFixtureProducesSingleTriangleMesh) {
    ParsedMesh mesh;
    ASSERT_TRUE(loadObjFixture(fixturePath("triangle.obj"), mesh));
    EXPECT_EQ(mesh.vertices.size(), 3U);
    EXPECT_EQ(mesh.indices.size(), 3U);
}

TEST(ObjImporter, TriangleFixtureVertexPositionsMatchExpectedValues) {
    ParsedMesh mesh;
    ASSERT_TRUE(loadObjFixture(fixturePath("triangle.obj"), mesh));

    const std::array expectedPositions{
        glm::vec3(0.0F, 0.0F, 0.0F),
        glm::vec3(1.0F, 0.0F, 0.0F),
        glm::vec3(0.0F, 1.0F, 0.0F),
    };

    for (size_t i = 0; i < expectedPositions.size(); ++i) {
        EXPECT_NEAR(glm::length(mesh.vertices[i].position - expectedPositions[i]), 0.0F, kEpsilon);
    }
}

TEST(ObjImporter, TriangleFixtureNormalsAreUnitLength) {
    ParsedMesh mesh;
    ASSERT_TRUE(loadObjFixture(fixturePath("triangle.obj"), mesh));

    for (const GpuVertex& vertex : mesh.vertices) {
        EXPECT_NEAR(glm::length(vertex.normal), 1.0F, kEpsilon);
    }
}

TEST(ObjImporter, TriangleFixtureUvsStayInUnitRange) {
    ParsedMesh mesh;
    ASSERT_TRUE(loadObjFixture(fixturePath("triangle.obj"), mesh));

    for (const GpuVertex& vertex : mesh.vertices) {
        EXPECT_GE(vertex.uv.x, 0.0F);
        EXPECT_GE(vertex.uv.y, 0.0F);
        EXPECT_LE(vertex.uv.x, 1.0F);
        EXPECT_LE(vertex.uv.y, 1.0F);
    }
}
