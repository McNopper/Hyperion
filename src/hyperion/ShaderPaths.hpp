#ifndef HYPERION_SHADERPATHS_HPP
#define HYPERION_SHADERPATHS_HPP

#include <filesystem>

#include "harmonia/renderer/Pipeline.hpp"

[[nodiscard]] inline Pipeline::ShaderPaths makeHyperionShaderPaths(const std::filesystem::path& rootDir) {
    return Pipeline::ShaderPaths{
        .raygen = rootDir / "raygen.spv",
        .closesthitTriangle = rootDir / "closesthit.spv",
        .closesthitSphere = rootDir / "closesthit.spv",
        .intersection = rootDir / "intersection.spv",
        .miss = rootDir / "miss.spv",
        .shadowMiss = rootDir / "shadow_miss.spv",
    };
}

#endif // HYPERION_SHADERPATHS_HPP
