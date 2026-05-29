// Trivial smoke test — no Vulkan, no SDL calls.
// If this crashes, the problem is in static initializers or DLL load, not test logic.
#include <gtest/gtest.h>

TEST(Smoke, TrueIsTrue) {
    ASSERT_TRUE(true);
}
