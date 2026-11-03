#include <OpenImageIO/imagebuf.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "demo/Application.hpp"

namespace {
class CaptureCase {
  public:
    explicit CaptureCase(bool disableAccumulation = true) {
        const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("hyperion_capture_contract_" + std::to_string(suffix));
        if (!std::filesystem::create_directory(root))
            return;
        const std::filesystem::path assets(HYPERION_ASSETS_DIR);
        std::ifstream original(assets / "cornell_classic.scene.toml");
        std::string text((std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
        // Absolute references keep this owned temporary fixture independent of CWD.
        for (const auto* ref :
             {"cornell.materials.toml", "presets/preview.render.toml", "presets/cornell.camera.toml", "cornell.obj"}) {
            const auto quoted = "\"" + std::string(ref) + "\"";
            const auto at = text.find(quoted);
            if (at != std::string::npos)
                text.replace(at, quoted.size(), "\"" + (assets / ref).generic_string() + "\"");
        }
        const auto header = text.find("[render]");
        if (disableAccumulation && header != std::string::npos)
            text.insert(text.find('\n', header) + 1,
                        "enable_accumulation_stage = false\nenable_denoiser_stage = false\n");
        scene = root / "disabled.scene.toml";
        output = root / "capture.exr";
        std::ofstream file(scene);
        file << text;
    }
    ~CaptureCase() {
        std::error_code ec;
        for (const auto& file : {scene, output, root / "capture.png"})
            std::filesystem::remove(file, ec);
        std::filesystem::remove(root, ec);
    }
    int run(std::uint32_t spp, std::string& log, std::uint32_t seed = 17) const {
        harmonia::app::AppConfig cfg;
        cfg.width = cfg.height = 8;
        cfg.assetsDir = std::filesystem::path(HYPERION_ASSETS_DIR);
        cfg.sceneFile = scene;
        cfg.outputFile = output;
        cfg.validation = true;
        cfg.noPostfx = true;
        cfg.rngSeed = seed;
        hyperion::Application::DemoConfig demo;
        demo.spp = spp;
        demo.sppExplicit = true;
        demo.shaderDir = std::filesystem::path(HYPERION_SHADER_DIR);
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        int result;
        {
            hyperion::Application app;
            result = app.run(std::move(cfg), std::move(demo));
        }
        log = testing::internal::GetCapturedStdout() + testing::internal::GetCapturedStderr();
        return result;
    }
    std::filesystem::path root, scene, output;
};
} // namespace

TEST(CaptureSamplingContract, MultiSampleCaptureRejectsDisabledAccumulation) {
    CaptureCase files;
    std::string log;
    EXPECT_NE(files.run(4, log), 0);
    EXPECT_NE(log.find("Hyperion multi-sample captures require the accumulation stage"), std::string::npos) << log;
    EXPECT_FALSE(std::filesystem::exists(files.output));
    EXPECT_FALSE(std::filesystem::exists(files.root / "capture.png"));
    EXPECT_EQ(log.find("VUID-"), std::string::npos) << log;
}

TEST(CaptureSamplingContract, SingleSampleCaptureDoesNotNeedTemporalHistory) {
    CaptureCase files;
    std::string log;
    EXPECT_EQ(files.run(1, log), 0) << log;
    EXPECT_TRUE(std::filesystem::exists(files.output));
    EXPECT_TRUE(std::filesystem::exists(files.root / "capture.png"));
    EXPECT_EQ(log.find("VUID-"), std::string::npos) << log;
}

TEST(CaptureSamplingContract, CaptureBudgetProgressiveStreamsAndCliSeedsAreHonored) {
    CaptureCase files(false);
    std::string log;
    auto pixels = [&files]() {
        OIIO::ImageBuf image(files.output.string());
        EXPECT_TRUE(image.read());
        std::vector<float> result(static_cast<std::size_t>(image.spec().width) * image.spec().height * 3);
        std::size_t index = 0;
        for (int y = 0; y < image.spec().height; ++y) {
            for (int x = 0; x < image.spec().width; ++x) {
                image.getpixel(x, y, result.data() + index, 3);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    EXPECT_TRUE(std::isfinite(result[index + channel]));
                }
                index += 3;
            }
        }
        return result;
    };
    ASSERT_EQ(files.run(8, log, 17), 0) << log;
    EXPECT_EQ(log.find("VUID-"), std::string::npos) << log;
    EXPECT_NE(log.find("Sampling plan: 8 frame(s), 1 sample(s)/dispatch (requested spp 8)"), std::string::npos) << log;
    const auto first = pixels();
    ASSERT_FALSE(first.empty());
    ASSERT_EQ(files.run(8, log, 17), 0) << log;
    EXPECT_EQ(log.find("VUID-"), std::string::npos) << log;
    EXPECT_EQ(first, pixels()) << "Same-seed captures must replay exactly";
    ASSERT_EQ(files.run(8, log, 91), 0) << log;
    EXPECT_EQ(log.find("VUID-"), std::string::npos) << log;
    EXPECT_NE(first, pixels()) << "The CLI seed must select a different stream";
    ASSERT_EQ(files.run(1, log, 17), 0) << log;
    EXPECT_EQ(log.find("VUID-"), std::string::npos) << log;
    EXPECT_NE(first, pixels()) << "Accumulating eight progressive samples must not repeat sample zero";
}
