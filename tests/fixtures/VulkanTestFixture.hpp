#pragma once

// Shared Vulkan test infrastructure for component and module tests.
//
// Usage:
//   1. In main.cpp: call setupVulkanTestContext() before RUN_ALL_TESTS(),
//      and teardownVulkanTestContext() after.
//   2. Derive your test fixture from VulkanFixture (basic Vulkan) or
//      RtFixture (also checks RT extensions are present).
//
// One Context is created per test binary; tests share it for speed.
// TearDown() calls vkDeviceWaitIdle so each test starts with a quiescent GPU.

#include <memory>

#include <volk/volk.h>

#include <gtest/gtest.h>

#include "demo/vulkan_init/Context.hpp"
#include "demo/vulkan_init/PhysicalDevice.hpp"
#include "hyperion/core/CommandPool.hpp"

struct VulkanTestContext {
    SDL_Window* window{};
    std::unique_ptr<Context> context;
    std::unique_ptr<CommandPool> commandPool;

    [[nodiscard]] bool isValid() const noexcept { return context && commandPool; }
    [[nodiscard]] const DeviceContext& deviceCtx() const noexcept { return context->deviceContext(); }
    [[nodiscard]] const PhysicalDeviceInfo& physInfo() const noexcept {
        return context->physicalDeviceInfo();
    }
};

// Set in main() before RUN_ALL_TESTS(); nullptr means Vulkan unavailable.
inline VulkanTestContext* g_vulkanTestCtx = nullptr;

// Base fixture: requires a valid Vulkan context (which in this project always
// includes RT because Context::create() selects only RT-capable devices).
class VulkanFixture : public ::testing::Test {
  protected:
    void SetUp() override {
        if (g_vulkanTestCtx == nullptr || !g_vulkanTestCtx->isValid()) {
            GTEST_SKIP() << "Vulkan context unavailable";
        }
    }

    void TearDown() override {
        if (g_vulkanTestCtx != nullptr && g_vulkanTestCtx->isValid()) {
            vkDeviceWaitIdle(g_vulkanTestCtx->deviceCtx().device);
        }
    }

    [[nodiscard]] const DeviceContext& deviceCtx() const noexcept {
        return g_vulkanTestCtx->deviceCtx();
    }
    [[nodiscard]] CommandPool& commandPool() const noexcept { return *g_vulkanTestCtx->commandPool; }
};

// Extended fixture: also skips if RT extension functions are not loaded.
// Use this for tests that exercise AccelerationStructure / SBT / PathTracer.
class RtFixture : public VulkanFixture {
  protected:
    void SetUp() override {
        VulkanFixture::SetUp();
        if (vkCmdTraceRaysKHR == nullptr) {
            GTEST_SKIP() << "Ray tracing not available on this device";
        }
    }

    [[nodiscard]] const PhysicalDeviceInfo& physInfo() const noexcept {
        return g_vulkanTestCtx->physInfo();
    }
};
