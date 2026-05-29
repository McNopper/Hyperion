#include "hyperion/scene/IblProbe.hpp"

#include <utility>

#include "hyperion/core/Buffer.hpp"
#include "hyperion/core/Logger.hpp"

#ifdef HYPERION_HAS_OPENEXR
#include <Imath/ImathBox.h>
#include <OpenEXR/ImfRgbaFile.h>
#endif

IblProbe::IblProbe(IblProbe&& other) noexcept
    : m_image(std::move(other.m_image)), m_sampler(std::exchange(other.m_sampler, VK_NULL_HANDLE)), m_ctx(other.m_ctx) {
    other.m_ctx = nullptr;
}

IblProbe& IblProbe::operator=(IblProbe&& other) noexcept {
    if (this != &other) {
        reset();
        m_image = std::move(other.m_image);
        m_sampler = std::exchange(other.m_sampler, VK_NULL_HANDLE);
        m_ctx = other.m_ctx;
        other.m_ctx = nullptr;
    }
    return *this;
}

IblProbe::~IblProbe() {
    reset();
}

void IblProbe::reset() noexcept {
    m_image = {};
    if (m_ctx != nullptr && m_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_ctx->device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    m_ctx = nullptr;
}

std::expected<IblProbe, VkResult>
IblProbe::loadFromEXR(const DeviceContext& ctx, const CommandPool& pool, const std::filesystem::path& path) {
#ifndef HYPERION_HAS_OPENEXR
    (void)ctx;
    (void)pool;
    (void)path;
    Logger::error("IblProbe: OpenEXR support is not compiled in; cannot load '{}'", path.string());
    return std::unexpected(VK_ERROR_FEATURE_NOT_PRESENT);
#else
    // ── Load EXR pixels ──────────────────────────────────────────────────────
    int width = 0, height = 0;
    std::vector<float> rgba32f;

    try {
        using namespace OPENEXR_IMF_NAMESPACE;
        RgbaInputFile file(path.string().c_str());
        const IMATH_NAMESPACE::Box2i dw = file.dataWindow();
        width = dw.max.x - dw.min.x + 1;
        height = dw.max.y - dw.min.y + 1;

        std::vector<Rgba> halfs(static_cast<size_t>(width * height));
        file.setFrameBuffer(halfs.data() - dw.min.x - dw.min.y * width, 1, width);
        file.readPixels(dw.min.y, dw.max.y);

        // ── Convert half RGBA → float RGBA, with lin_srgb → lin_rec2020 ──────
        // Rec.709 → Rec.2020 primary transform (D65 white point, IEC 61966 / BT.2087)
        const float m00 = 0.6274040f, m01 = 0.3292820f, m02 = 0.0433140f;
        const float m10 = 0.0690970f, m11 = 0.9195400f, m12 = 0.0113630f;
        const float m20 = 0.0163916f, m21 = 0.0880132f, m22 = 0.8955950f;

        rgba32f.resize(static_cast<size_t>(width * height) * 4u);
        for (int i = 0; i < width * height; ++i) {
            const float r = static_cast<float>(halfs[i].r);
            const float g = static_cast<float>(halfs[i].g);
            const float b = static_cast<float>(halfs[i].b);
            rgba32f[i * 4 + 0] = m00 * r + m01 * g + m02 * b;
            rgba32f[i * 4 + 1] = m10 * r + m11 * g + m12 * b;
            rgba32f[i * 4 + 2] = m20 * r + m21 * g + m22 * b;
            rgba32f[i * 4 + 3] = static_cast<float>(halfs[i].a);
        }
    } catch (const std::exception& e) {
        Logger::error("IblProbe: failed to read '{}': {}", path.string(), e.what());
        return std::unexpected(VK_ERROR_INITIALIZATION_FAILED);
    }

    // ── Upload to GPU ────────────────────────────────────────────────────────
    const VkExtent2D extent{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(width * height) * 4u * sizeof(float);

    auto staging = Buffer::create(
        ctx, byteSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, "ibl.staging");
    if (!staging) {
        return std::unexpected(staging.error());
    }
    staging->uploadData(rgba32f.data(), byteSize);

    auto image = Image::create(ctx,
                               extent,
                               VK_FORMAT_R32G32B32A32_SFLOAT,
                               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               "ibl.env");
    if (!image) {
        return std::unexpected(image.error());
    }

    auto cmd = pool.beginOneShot();
    if (!cmd) {
        return std::unexpected(cmd.error());
    }

    image->transition(*cmd,
                      VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_PIPELINE_STAGE_2_NONE,
                      0,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT);

    const VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {extent.width, extent.height, 1u},
    };
    vkCmdCopyBufferToImage(*cmd, staging->handle(), image->handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    image->transition(*cmd,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COPY_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                      VK_ACCESS_2_SHADER_READ_BIT);

    if (const VkResult result = pool.endOneShot(*cmd); result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    // ── Create sampler (REPEAT on U, CLAMP_TO_EDGE on V to avoid pole artefacts) ──
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler sampler = VK_NULL_HANDLE;
    if (const VkResult result = vkCreateSampler(ctx.device, &samplerInfo, nullptr, &sampler); result != VK_SUCCESS) {
        return std::unexpected(result);
    }

    IblProbe probe;
    probe.m_ctx = &ctx;
    probe.m_image = std::move(*image);
    probe.m_sampler = sampler;
    Logger::info("IblProbe: loaded '{}' ({}×{})", path.filename().string(), width, height);
    return probe;
#endif
}
