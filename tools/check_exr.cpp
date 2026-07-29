#include <Imath/ImathBox.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc, char* argv[]) {
    const char* path = argc > 1 ? argv[1] : "assets/meadow_2_4k.exr";
    using namespace OPENEXR_IMF_NAMESPACE;
    RgbaInputFile file(path);
    const IMATH_NAMESPACE::Box2i dw = file.dataWindow();
    const int width = dw.max.x - dw.min.x + 1;
    const int height = dw.max.y - dw.min.y + 1;
    std::vector<Rgba> pixels(static_cast<std::size_t>(width * height));
    file.setFrameBuffer(pixels.data() - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
    long long nanCount = 0, infCount = 0, negCount = 0;
    float maxFinite = 0.0f;
    for (const auto& px : pixels) {
        for (float v : {(float)px.r, (float)px.g, (float)px.b}) {
            if (std::isnan(v))
                ++nanCount;
            else if (std::isinf(v))
                ++infCount;
            else if (v < 0.0f)
                ++negCount;
            else
                maxFinite = std::max(maxFinite, v);
        }
    }
    std::printf("Size      : %d x %d\n", width, height);
    std::printf("NaN       : %lld\n", nanCount);
    std::printf("Inf       : %lld\n", infCount);
    std::printf("< 0       : %lld\n", negCount);
    std::printf("MaxFinite : %.4f\n", maxFinite);
    return 0;
}
