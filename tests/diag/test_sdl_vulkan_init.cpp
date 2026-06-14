// Diagnostic: tests SDL3 + Vulkan init step by step to pinpoint crashes.
// Each step prints a line before and after so we can see exactly where it stops.

#include <cstdio>
#include <cstdlib>

// ---- Step 1: can we even load volk headers? --------------------------------
#include <volk/volk.h>

// ---- Step 2: can we load SDL3 headers? ------------------------------------
// SDL_MAIN_HANDLED tells SDL_main.h not to redefine main; we handle it ourselves.
#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>

int main(int /*argc*/, char* /*argv*/[]) {
    std::puts("DIAG: process started");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step A: volk init
    // ------------------------------------------------------------------
    std::puts("DIAG: calling volkInitialize...");
    std::fflush(stdout);
    const VkResult volkResult = volkInitialize();
    if (volkResult != VK_SUCCESS) {
        std::printf("DIAG: volkInitialize FAILED (VkResult=%d) — Vulkan loader not found\n",
                    static_cast<int>(volkResult));
        return 1;
    }
    std::puts("DIAG: volkInitialize OK");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step B: SDL_SetMainReady (required when not going through SDL_main)
    // ------------------------------------------------------------------
    std::puts("DIAG: calling SDL_SetMainReady...");
    std::fflush(stdout);
    SDL_SetMainReady();
    std::puts("DIAG: SDL_SetMainReady OK");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step C: SDL_Init
    // ------------------------------------------------------------------
    std::puts("DIAG: calling SDL_Init(SDL_INIT_VIDEO)...");
    std::fflush(stdout);
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::printf("DIAG: SDL_Init FAILED: %s\n", SDL_GetError());
        return 1;
    }
    std::puts("DIAG: SDL_Init OK");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step D: SDL_CreateWindow (hidden, Vulkan)
    // ------------------------------------------------------------------
    std::puts("DIAG: calling SDL_CreateWindow...");
    std::fflush(stdout);
    SDL_Window* window = SDL_CreateWindow("diag", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
    if (!window) {
        std::printf("DIAG: SDL_CreateWindow FAILED: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    std::puts("DIAG: SDL_CreateWindow OK");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step E: SDL_Vulkan_GetInstanceExtensions
    // ------------------------------------------------------------------
    std::puts("DIAG: calling SDL_Vulkan_GetInstanceExtensions...");
    std::fflush(stdout);
    Uint32 extCount = 0;
    const char* const* exts = SDL_Vulkan_GetInstanceExtensions(&extCount);
    if (!exts || extCount == 0) {
        std::printf("DIAG: SDL_Vulkan_GetInstanceExtensions FAILED: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    std::printf("DIAG: SDL_Vulkan_GetInstanceExtensions OK — %u extensions\n", extCount);
    for (Uint32 i = 0; i < extCount; ++i) {
        std::printf("  [%u] %s\n", i, exts[i]);
    }
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step F: vkCreateInstance
    // ------------------------------------------------------------------
    std::puts("DIAG: creating VkInstance...");
    std::fflush(stdout);
    const VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "hyperion-diag",
        .applicationVersion = 1,
        .pEngineName = "Hyperion",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_4,
    };
    const VkInstanceCreateInfo instInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = extCount,
        .ppEnabledExtensionNames = exts,
    };
    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&instInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        std::printf("DIAG: vkCreateInstance FAILED (VkResult=%d)\n", static_cast<int>(result));
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    volkLoadInstance(instance);
    std::puts("DIAG: vkCreateInstance OK");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step G: SDL_Vulkan_CreateSurface
    // ------------------------------------------------------------------
    std::puts("DIAG: calling SDL_Vulkan_CreateSurface...");
    std::fflush(stdout);
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        std::printf("DIAG: SDL_Vulkan_CreateSurface FAILED: %s\n", SDL_GetError());
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    std::puts("DIAG: SDL_Vulkan_CreateSurface OK");
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Step H: enumerate physical devices
    // ------------------------------------------------------------------
    std::puts("DIAG: enumerating physical devices...");
    std::fflush(stdout);
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::printf("DIAG: found %u physical device(s)\n", deviceCount);
    std::fflush(stdout);

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::puts("DIAG: all steps passed");
    return 0;
}
