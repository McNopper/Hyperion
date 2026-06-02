#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "demo/importers/MaterialLibrary.hpp"
#include "hyperion/GpuTypes.hpp"

namespace {

constexpr float kEps = 1.0e-4F;

[[nodiscard]] std::filesystem::path fixtureMtlx() {
    return std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / "openpbr_test.mtlx";
}

class MaterialLibraryTest : public ::testing::Test {
  protected:
    void SetUp() override { ASSERT_TRUE(m_lib.load(fixtureMtlx())); }

    MaterialLibrary m_lib;
};

} // namespace

// ── Load ─────────────────────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, LoadsSevenMaterials) {
    EXPECT_EQ(m_lib.size(), 7U);
}

TEST_F(MaterialLibraryTest, GetReturnsNulloptForUnknownName) {
    EXPECT_FALSE(m_lib.get("NonExistent").has_value());
}

TEST_F(MaterialLibraryTest, GetOrDefaultReturnsMaterialForUnknownName) {
    const Material mat = m_lib.getOrDefault("NonExistent");
    // Should return a valid (non-zero) material without crashing.
    EXPECT_GE(mat.gpu().baseColorWeight.w, 0.0F);
}

// ── DiffuseGray ───────────────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, DiffuseGrayIsPresent) {
    EXPECT_TRUE(m_lib.get("DiffuseGray").has_value());
}

TEST_F(MaterialLibraryTest, DiffuseGrayRoughness) {
    const GpuMaterial g = m_lib.get("DiffuseGray")->gpu();
    EXPECT_NEAR(g.specularRoughAnisoIor.x, 0.80F, kEps);
}

TEST_F(MaterialLibraryTest, DiffuseGrayMetalnessIsZero) {
    const GpuMaterial g = m_lib.get("DiffuseGray")->gpu();
    EXPECT_NEAR(g.baseMetalnessDiffRough.x, 0.0F, kEps);
}

TEST_F(MaterialLibraryTest, DiffuseGrayTransmissionIsZero) {
    const GpuMaterial g = m_lib.get("DiffuseGray")->gpu();
    EXPECT_NEAR(g.transmissionColorWeight.w, 0.0F, kEps);
}

// ── Copper (conductor) ────────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, CopperIsPresent) {
    EXPECT_TRUE(m_lib.get("Copper").has_value());
}

TEST_F(MaterialLibraryTest, CopperMetalnessIsOne) {
    const GpuMaterial g = m_lib.get("Copper")->gpu();
    EXPECT_NEAR(g.baseMetalnessDiffRough.x, 1.0F, kEps);
}

TEST_F(MaterialLibraryTest, CopperRoughness) {
    const GpuMaterial g = m_lib.get("Copper")->gpu();
    EXPECT_NEAR(g.specularRoughAnisoIor.x, 0.02F, kEps);
}

TEST_F(MaterialLibraryTest, CopperBaseColorIsNonZero) {
    // base_color is converted from lin_rec709 → lin_rec2020, so the values will
    // differ from the raw 0.811/0.643/0.542 inputs — but they must remain > 0.
    const GpuMaterial g = m_lib.get("Copper")->gpu();
    EXPECT_GT(g.baseColorWeight.x, 0.0F);
    EXPECT_GT(g.baseColorWeight.y, 0.0F);
    EXPECT_GT(g.baseColorWeight.z, 0.0F);
}

TEST_F(MaterialLibraryTest, CopperSpecularColorIsNonZero) {
    // F82 term: specular_color should be converted and stored.
    const GpuMaterial g = m_lib.get("Copper")->gpu();
    EXPECT_GT(g.specularColorWeight.x, 0.0F);
    EXPECT_GT(g.specularColorWeight.y, 0.0F);
    EXPECT_GT(g.specularColorWeight.z, 0.0F);
}

TEST_F(MaterialLibraryTest, CopperFlagsNotGlass) {
    const GpuMaterial g = m_lib.get("Copper")->gpu();
    EXPECT_NEAR(g.opacityFlagsPad.y, 0.0F, kEps);
}

// ── Glass (dielectric) ────────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, GlassIsPresent) {
    EXPECT_TRUE(m_lib.get("Glass").has_value());
}

TEST_F(MaterialLibraryTest, GlassTransmissionWeightIsOne) {
    const GpuMaterial g = m_lib.get("Glass")->gpu();
    EXPECT_NEAR(g.transmissionColorWeight.w, 1.0F, kEps);
}

TEST_F(MaterialLibraryTest, GlassIOR) {
    const GpuMaterial g = m_lib.get("Glass")->gpu();
    EXPECT_NEAR(g.specularRoughAnisoIor.z, 1.52F, kEps);
}

TEST_F(MaterialLibraryTest, GlassRoughnessIsZero) {
    const GpuMaterial g = m_lib.get("Glass")->gpu();
    EXPECT_NEAR(g.specularRoughAnisoIor.x, 0.0F, kEps);
}

TEST_F(MaterialLibraryTest, GlassFlagsIsTwo) {
    // transmission_weight >= 0.5 and base_metalness == 0 → flags = 2 (glass mode)
    const GpuMaterial g = m_lib.get("Glass")->gpu();
    EXPECT_NEAR(g.opacityFlagsPad.y, 2.0F, kEps);
}

TEST_F(MaterialLibraryTest, GlassMetalnessIsZero) {
    const GpuMaterial g = m_lib.get("Glass")->gpu();
    EXPECT_NEAR(g.baseMetalnessDiffRough.x, 0.0F, kEps);
}

// ── Carpaint (clearcoat) ──────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, CarpaintIsPresent) {
    EXPECT_TRUE(m_lib.get("Carpaint").has_value());
}

TEST_F(MaterialLibraryTest, CarpaintCoatWeight) {
    const GpuMaterial g = m_lib.get("Carpaint")->gpu();
    EXPECT_NEAR(g.coatColorWeight.w, 1.0F, kEps);
}

TEST_F(MaterialLibraryTest, CarpaintCoatRoughness) {
    const GpuMaterial g = m_lib.get("Carpaint")->gpu();
    EXPECT_NEAR(g.coatRoughAnisoIorDark.x, 0.02F, kEps);
}

TEST_F(MaterialLibraryTest, CarpaintCoatIOR) {
    const GpuMaterial g = m_lib.get("Carpaint")->gpu();
    EXPECT_NEAR(g.coatRoughAnisoIorDark.z, 1.60F, kEps);
}

TEST_F(MaterialLibraryTest, CarpaintBaseIOR) {
    const GpuMaterial g = m_lib.get("Carpaint")->gpu();
    EXPECT_NEAR(g.specularRoughAnisoIor.z, 1.60F, kEps);
}

// ── Velvet (fuzz lobe) ────────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, VelvetIsPresent) {
    EXPECT_TRUE(m_lib.get("Velvet").has_value());
}

TEST_F(MaterialLibraryTest, VelvetFuzzWeight) {
    const GpuMaterial g = m_lib.get("Velvet")->gpu();
    EXPECT_NEAR(g.fuzzColorWeight.w, 0.5F, kEps);
}

TEST_F(MaterialLibraryTest, VelvetFuzzRoughness) {
    const GpuMaterial g = m_lib.get("Velvet")->gpu();
    EXPECT_NEAR(g.fuzzRoughPad.x, 0.5F, kEps);
}

TEST_F(MaterialLibraryTest, VelvetFuzzColorIsNonZero) {
    const GpuMaterial g = m_lib.get("Velvet")->gpu();
    EXPECT_GT(g.fuzzColorWeight.x + g.fuzzColorWeight.y + g.fuzzColorWeight.z, 0.0F);
}

TEST_F(MaterialLibraryTest, VelvetSpecularRoughnessIsOne) {
    const GpuMaterial g = m_lib.get("Velvet")->gpu();
    EXPECT_NEAR(g.specularRoughAnisoIor.x, 1.0F, kEps);
}

// ── SkinI (subsurface) ────────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, SkinIIsPresent) {
    EXPECT_TRUE(m_lib.get("SkinI").has_value());
}

TEST_F(MaterialLibraryTest, SkinISubsurfaceWeight) {
    const GpuMaterial g = m_lib.get("SkinI")->gpu();
    EXPECT_NEAR(g.subsurfaceColorWeight.w, 1.0F, kEps);
}

TEST_F(MaterialLibraryTest, SkinISubsurfaceColorIsNonZero) {
    const GpuMaterial g = m_lib.get("SkinI")->gpu();
    EXPECT_GT(g.subsurfaceColorWeight.x, 0.0F);
}

TEST_F(MaterialLibraryTest, SkinISubsurfaceRadiusIsNonZero) {
    const GpuMaterial g = m_lib.get("SkinI")->gpu();
    EXPECT_GT(g.subsurfaceRadiusScale.x, 0.0F);
    EXPECT_GT(g.subsurfaceRadiusScale.y, 0.0F);
    EXPECT_GT(g.subsurfaceRadiusScale.z, 0.0F);
}

TEST_F(MaterialLibraryTest, SkinISpecularIOR) {
    const GpuMaterial g = m_lib.get("SkinI")->gpu();
    EXPECT_NEAR(g.specularRoughAnisoIor.z, 1.40F, kEps);
}

// ── Emissive ──────────────────────────────────────────────────────────────────

TEST_F(MaterialLibraryTest, EmissiveIsPresent) {
    EXPECT_TRUE(m_lib.get("Emissive").has_value());
}

TEST_F(MaterialLibraryTest, EmissiveLuminance) {
    const GpuMaterial g = m_lib.get("Emissive")->gpu();
    EXPECT_NEAR(g.emissionColorLum.w, 100000.0F, kEps);
}

TEST_F(MaterialLibraryTest, EmissiveColorIsNonZero) {
    const GpuMaterial g = m_lib.get("Emissive")->gpu();
    EXPECT_GT(g.emissionColorLum.x + g.emissionColorLum.y + g.emissionColorLum.z, 0.0F);
}

// ── File-level colorspace ─────────────────────────────────────────────────────

TEST(MaterialLibraryColorspace, Lin709InputIsConvertedToRec2020) {
    // A pure-red color in lin_rec709 (1,0,0) should produce a different
    // value in lin_rec2020 — specifically it gets a green component.
    const std::string content = "colorspace lin_rec709\n"
                                "newmtl PureRed709\n"
                                "base_color 1.0 0.0 0.0\n";

    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "hyperion_test_cs.mtlx";
    {
        std::ofstream out(tmp);
        out << content;
    }

    MaterialLibrary lib;
    ASSERT_TRUE(lib.load(tmp));
    const GpuMaterial& g = lib.get("PureRed709")->gpu();
    // In Rec.2020 a pure-red Rec.709 primary has a small but non-zero green channel.
    EXPECT_GT(g.baseColorWeight.y, 0.0F);
    std::filesystem::remove(tmp);
}

TEST(MaterialLibraryColorspace, Lin2020InputIsNotConverted) {
    const std::string content = "colorspace lin_rec2020\n"
                                "newmtl PureRed2020\n"
                                "base_color 1.0 0.0 0.0\n";

    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "hyperion_test_cs2.mtlx";
    {
        std::ofstream out(tmp);
        out << content;
    }

    MaterialLibrary lib;
    ASSERT_TRUE(lib.load(tmp));
    const GpuMaterial& g = lib.get("PureRed2020")->gpu();
    EXPECT_NEAR(g.baseColorWeight.x, 1.0F, kEps);
    EXPECT_NEAR(g.baseColorWeight.y, 0.0F, kEps);
    EXPECT_NEAR(g.baseColorWeight.z, 0.0F, kEps);
    std::filesystem::remove(tmp);
}

// ── Edge cases ────────────────────────────────────────────────────────────────

TEST(MaterialLibraryEdge, EmptyFileLoadsSuccessfully) {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "hyperion_empty.mtlx";
    {
        std::ofstream out(tmp);
    }
    MaterialLibrary lib;
    EXPECT_TRUE(lib.load(tmp));
    EXPECT_TRUE(lib.empty());
    std::filesystem::remove(tmp);
}

TEST(MaterialLibraryEdge, NonExistentFileReturnsFalse) {
    MaterialLibrary lib;
    EXPECT_FALSE(lib.load("/no/such/file.mtlx"));
}

TEST(MaterialLibraryEdge, CommentsAreIgnored) {
    const std::string content = "# This is a comment\n"
                                "newmtl Commented\n"
                                "# Another comment\n"
                                "base_color 0.5 0.5 0.5  # inline comment\n"
                                "specular_roughness 0.4\n";

    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "hyperion_comments.mtlx";
    {
        std::ofstream out(tmp);
        out << content;
    }

    MaterialLibrary lib;
    ASSERT_TRUE(lib.load(tmp));
    EXPECT_EQ(lib.size(), 1U);
    EXPECT_TRUE(lib.get("Commented").has_value());
    EXPECT_NEAR(lib.get("Commented")->gpu().specularRoughAnisoIor.x, 0.4F, kEps);
    std::filesystem::remove(tmp);
}

TEST(MaterialLibraryEdge, ClassicMtlKeywordsAreSilentlyIgnored) {
    // Kd, Ks, Pm, Pr, Ni, Tr, Ke, map_Kd, map_Ns, map_bump are classic OBJ/MTL
    // keywords — they must be silently ignored. Proof: Kd sets green=0.2, blue=0.1;
    // if ignored those channels stay at the default base_color value (0.8), not 0.2/0.1.
    const std::string content = "colorspace lin_rec709\n"
                                "newmtl ClassicMat\n"
                                "Kd 0.8 0.2 0.1\n"
                                "Ks 1.0 1.0 1.0\n"
                                "Pm 1.0\n"
                                "Pr 0.3\n"
                                "Ni 1.5\n"
                                "Tr 1.0\n"
                                "Ke 5.0 4.0 3.0\n"
                                "map_Kd some_texture.png\n"
                                "map_Ns roughness.png\n"
                                "map_bump normal.png\n";

    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "hyperion_classic.mtlx";
    {
        std::ofstream out(tmp);
        out << content;
    }

    MaterialLibrary lib;
    ASSERT_TRUE(lib.load(tmp));
    ASSERT_TRUE(lib.get("ClassicMat").has_value());
    const GpuMaterial g = lib.get("ClassicMat")->gpu();
    // Kd green=0.2, blue=0.1 must NOT be applied — channels stay at default 0.8
    EXPECT_GT(g.baseColorWeight.y, 0.5F) << "Kd green (0.2) must not override default base_color green";
    EXPECT_GT(g.baseColorWeight.z, 0.5F) << "Kd blue (0.1) must not override default base_color blue";
    // transmission_weight stays 0 — Tr was ignored
    EXPECT_NEAR(g.transmissionColorWeight.w, 0.0F, kEps);
    // emission luminance stays 0 — Ke was ignored
    EXPECT_NEAR(g.emissionColorLum.w, 0.0F, kEps);
    // map_Kd / map_Ns / map_bump must not be recognised — texture indices stay sentinel
    EXPECT_EQ(g.textureIndices.x, kNoTexture) << "map_Kd must not populate base_color texture slot";
    EXPECT_EQ(g.textureIndices.y, kNoTexture) << "map_bump must not populate normal texture slot";
    std::filesystem::remove(tmp);
}
