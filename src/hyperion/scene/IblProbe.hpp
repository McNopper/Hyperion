#pragma once

#include <volk/volk.h>

#include <expected>
#include <filesystem>

#include "hyperion/DeviceContext.hpp"
#include "hyperion/core/CommandPool.hpp"
#include "hyperion/core/Image.hpp"

/// Image-based lighting probe loaded from an equirectangular HDR panorama (EXR).
///
/// The panorama is stored on the GPU as a 2D RGBA32F texture in linear Rec.2020.
/// Input EXR files are assumed to be in linear sRGB (Rec.709 primaries, D65)
/// and are converted to linear Rec.2020 at load time.
class IblProbe {
  public:
    IblProbe() = default;
    IblProbe(const IblProbe&) = delete;
    IblProbe& operator=(const IblProbe&) = delete;
    IblProbe(IblProbe&& other) noexcept;
    IblProbe& operator=(IblProbe&& other) noexcept;
    ~IblProbe();

    /// Load an equirectangular EXR panorama, convert to linear Rec.2020, and upload to GPU.
    /// Requires HYPERION_HAS_OPENEXR; returns VK_ERROR_FEATURE_NOT_PRESENT otherwise.
    [[nodiscard]] static std::expected<IblProbe, VkResult>
    loadFromEXR(const DeviceContext& ctx, const CommandPool& pool, const std::filesystem::path& path);

    [[nodiscard]] VkImageView imageView() const noexcept { return m_image.view(); }
    [[nodiscard]] VkSampler sampler() const noexcept { return m_sampler; }
    [[nodiscard]] bool isValid() const noexcept { return m_sampler != VK_NULL_HANDLE; }

  private:
    void reset() noexcept;

    Image m_image{};
    VkSampler m_sampler{VK_NULL_HANDLE};
    const DeviceContext* m_ctx{};
};
