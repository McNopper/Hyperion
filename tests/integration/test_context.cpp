#include <volk/volk.h>
#include <SDL3/SDL.h>

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "harmonia/vulkan_init/Context.hpp"

namespace {
class SdlVideoScope {
  public:
    SdlVideoScope() noexcept : m_initialized(SDL_Init(SDL_INIT_VIDEO)) {}

    ~SdlVideoScope() {
        if (m_initialized) {
            SDL_Quit();
        }
    }

    [[nodiscard]] bool initialized() const noexcept { return m_initialized; }

  private:
    bool m_initialized = false;
};

struct WindowDeleter {
    void operator()(SDL_Window* window) const noexcept {
        if (window != nullptr) {
            SDL_DestroyWindow(window);
        }
    }
};
} // namespace

TEST(Context, CreateContext) {
    if (const VkResult volkResult = volkInitialize(); volkResult != VK_SUCCESS) {
        GTEST_SKIP() << "Vulkan loader unavailable: VkResult=" << static_cast<int>(volkResult);
    }

    SdlVideoScope sdl;
    if (!sdl.initialized()) {
        GTEST_SKIP() << "SDL video initialization failed: " << SDL_GetError();
    }

    std::unique_ptr<SDL_Window, WindowDeleter> window(
        SDL_CreateWindow("Hyperion Context Test", 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN));
    if (!window) {
        GTEST_SKIP() << "Failed to create Vulkan test window: " << SDL_GetError();
    }

    Context::Config config{};
    config.appName = "HyperionTestContext";
    config.enableValidation = false;
    config.window = window.get();

    auto context = Context::create(config);
    if (!context) {
        GTEST_SKIP() << "No Vulkan RT-capable device/context available: VkResult=" << static_cast<int>(context.error());
    }

    const DeviceContext& deviceContext = context->deviceContext();
    EXPECT_NE(deviceContext.device, VK_NULL_HANDLE);
    EXPECT_NE(deviceContext.physicalDevice, VK_NULL_HANDLE);
    EXPECT_NE(deviceContext.graphicsQueue, VK_NULL_HANDLE);
    // volk globals are loaded by volkLoadDevice inside Context::create.
    EXPECT_NE(vkCreateAccelerationStructureKHR, nullptr);
    EXPECT_NE(vkCmdTraceRaysKHR, nullptr);
    EXPECT_NE(vkCmdPushDescriptorSet, nullptr);
}
