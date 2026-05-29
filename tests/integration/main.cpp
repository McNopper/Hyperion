// Custom main for integration tests.
// SDL3 requires SDL_SetMainReady() to be called before SDL_Init() whenever
// the program does not use SDL's own entry point (SDL_main / SDL_RunApp).
// Since GTest owns main(), we provide our own main here and link against
// GTest::gtest (not GTest::gtest_main).

// SDL_MAIN_HANDLED tells SDL_main.h not to redefine main() — we own it.
#define SDL_MAIN_HANDLED
#include <SDL3/SDL_main.h>

#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
    // Must happen before any SDL call, including SDL_Init inside tests.
    SDL_SetMainReady();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
