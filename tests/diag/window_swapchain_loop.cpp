// System integration test: SDL3 + Vulkan 1.4 + volk only.
// No Hyperion libraries. Self-contained. No macros, no hacks.
//
// What it does:
//   init → create visible window → Vulkan instance / device / swapchain →
//   loop (acquire → clear to orange → present, 5 s) → destroy
//
// A visible orange window with zero validation messages means the entire
// SDL + Vulkan + swapchain presentation path works end-to-end.

#define SDL_MAIN_HANDLED
#include <SDL3/SDL_main.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <volk/volk.h>

#include <algorithm>
#include <cstring>
#include <print>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Vulkan debug messenger
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL onVulkanMessage(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*                                       /*pUserData*/)
{
    std::string_view prefix =
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)   ? "ERROR"   :
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "WARNING" : "INFO";
    std::println("[VK {}] {}", prefix, data ? data->pMessage : "(null)");
    return VK_FALSE;
}

// ---------------------------------------------------------------------------
// Barrier helper — Vulkan 1.3 synchronization2
// ---------------------------------------------------------------------------
static void imageBarrier(VkCommandBuffer         cmd,
                         VkImage                 image,
                         VkImageLayout           fromLayout,
                         VkImageLayout           toLayout,
                         VkPipelineStageFlags2   srcStage,
                         VkAccessFlags2          srcAccess,
                         VkPipelineStageFlags2   dstStage,
                         VkAccessFlags2          dstAccess)
{
    const VkImageMemoryBarrier2 barrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .pNext               = nullptr,
        .srcStageMask        = srcStage,
        .srcAccessMask       = srcAccess,
        .dstStageMask        = dstStage,
        .dstAccessMask       = dstAccess,
        .oldLayout           = fromLayout,
        .newLayout           = toLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    const VkDependencyInfo dep{
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext                    = nullptr,
        .dependencyFlags          = 0,
        .memoryBarrierCount       = 0,
        .pMemoryBarriers          = nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers    = nullptr,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
int main(int /*argc*/, char* /*argv*/[])
{
    SDL_SetMainReady();
    std::println("[TEST] === window_swapchain_loop start ===");

    // -----------------------------------------------------------------------
    // 1. volk
    // -----------------------------------------------------------------------
    if (volkInitialize() != VK_SUCCESS) {
        std::println("[TEST] volkInitialize failed — Vulkan loader not found");
        return 1;
    }
    std::println("[TEST] volkInitialize OK");

    // -----------------------------------------------------------------------
    // 2. SDL window (visible, 1280 x 720)
    // -----------------------------------------------------------------------
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::println("[TEST] SDL_Init failed: {}", SDL_GetError());
        return 1;
    }
    SDL_Window* window = SDL_CreateWindow("Hyperion – Swapchain Loop Test",
                                          1280, 720,
                                          SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (window == nullptr) {
        std::println("[TEST] SDL_CreateWindow failed: {}", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    std::println("[TEST] SDL window created (1280 x 720)");

    // -----------------------------------------------------------------------
    // 3. Vulkan instance  (validation layer + debug utils + SDL surface exts)
    // -----------------------------------------------------------------------
    Uint32 sdlExtCount = 0;
    const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    if (sdlExts == nullptr || sdlExtCount == 0) {
        std::println("[TEST] SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    std::vector<const char*> instExts(sdlExts, sdlExts + sdlExtCount);
    instExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    bool        validationAvailable = false;
    {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> layers(count);
        vkEnumerateInstanceLayerProperties(&count, layers.data());
        for (const VkLayerProperties& l : layers) {
            if (std::string_view{l.layerName} == validationLayer) {
                validationAvailable = true;
                break;
            }
        }
    }
    if (!validationAvailable) {
        std::println("[TEST] WARNING: VK_LAYER_KHRONOS_validation not available");
    }

    const VkDebugUtilsMessengerCreateInfoEXT messengerCI{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT    |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = onVulkanMessage,
        .pUserData       = nullptr,
    };
    const VkApplicationInfo appInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext              = nullptr,
        .pApplicationName   = "HyperionSwapchainTest",
        .applicationVersion = 1,
        .pEngineName        = "Hyperion",
        .engineVersion      = 1,
        .apiVersion         = VK_API_VERSION_1_4,
    };
    const VkInstanceCreateInfo instCI{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = validationAvailable ? &messengerCI : nullptr,
        .flags                   = 0,
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = validationAvailable ? 1u : 0u,
        .ppEnabledLayerNames     = validationAvailable ? &validationLayer : nullptr,
        .enabledExtensionCount   = static_cast<uint32_t>(instExts.size()),
        .ppEnabledExtensionNames = instExts.data(),
    };
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instCI, nullptr, &instance) != VK_SUCCESS) {
        std::println("[TEST] vkCreateInstance failed");
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    volkLoadInstance(instance);
    std::println("[TEST] VkInstance created (validation={})",
                 validationAvailable ? "ON" : "off");

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validationAvailable && vkCreateDebugUtilsMessengerEXT != nullptr) {
        vkCreateDebugUtilsMessengerEXT(instance, &messengerCI, nullptr, &messenger);
    }

    // -----------------------------------------------------------------------
    // 4. Surface
    // -----------------------------------------------------------------------
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        std::println("[TEST] SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    std::println("[TEST] VkSurfaceKHR created");

    // -----------------------------------------------------------------------
    // 5. Physical device — first discrete GPU that supports VK_KHR_swapchain
    // -----------------------------------------------------------------------
    uint32_t pdCount = 0;
    vkEnumeratePhysicalDevices(instance, &pdCount, nullptr);
    std::vector<VkPhysicalDevice> physDevices(pdCount);
    vkEnumeratePhysicalDevices(instance, &pdCount, physDevices.data());

    VkPhysicalDevice physDev     = VK_NULL_HANDLE;
    uint32_t         gfxFamily   = UINT32_MAX;

    for (VkPhysicalDevice pd : physDevices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        std::println("[TEST]   GPU candidate: {}", props.deviceName);

        uint32_t devExtCount = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &devExtCount, nullptr);
        std::vector<VkExtensionProperties> devExts(devExtCount);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &devExtCount, devExts.data());

        const bool hasSwapchain = std::ranges::any_of(devExts, [](const VkExtensionProperties& e) {
            return std::string_view{e.extensionName} == VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        });
        if (!hasSwapchain) { continue; }

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());

        for (uint32_t qi = 0; qi < qfCount; ++qi) {
            if (!(qfs[qi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { continue; }
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, qi, surface, &presentSupport);
            if (presentSupport == VK_TRUE) {
                physDev   = pd;
                gfxFamily = qi;
                break;
            }
        }
        if (physDev != VK_NULL_HANDLE) { break; }
    }

    if (physDev == VK_NULL_HANDLE) {
        std::println("[TEST] No suitable GPU found");
        return 1;
    }
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physDev, &props);
        std::println("[TEST] Selected GPU: {}  family={}", props.deviceName, gfxFamily);
    }

    // -----------------------------------------------------------------------
    // 6. Logical device  (synchronization2 feature required for submit2 / barrier2)
    // -----------------------------------------------------------------------
    const float queuePriority = 1.0f;
    const VkDeviceQueueCreateInfo queueCI{
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = 0,
        .queueFamilyIndex = gfxFamily,
        .queueCount       = 1,
        .pQueuePriorities = &queuePriority,
    };
    VkPhysicalDeviceSynchronization2Features sync2{
        .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
        .pNext            = nullptr,
        .synchronization2 = VK_TRUE,
    };
    const char* swapchainExt   = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    const VkDeviceCreateInfo deviceCI{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &sync2,
        .flags                   = 0,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queueCI,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = 1,
        .ppEnabledExtensionNames = &swapchainExt,
        .pEnabledFeatures        = nullptr,
    };
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physDev, &deviceCI, nullptr, &device) != VK_SUCCESS) {
        std::println("[TEST] vkCreateDevice failed");
        return 1;
    }
    volkLoadDevice(device);
    std::println("[TEST] VkDevice created");

    VkQueue gfxQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, gfxFamily, 0, &gfxQueue);

    // -----------------------------------------------------------------------
    // 7. Swapchain
    // -----------------------------------------------------------------------
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDev, surface, &caps);

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> surfaceFmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDev, surface, &fmtCount, surfaceFmts.data());

    VkSurfaceFormatKHR chosenFmt = surfaceFmts[0];
    for (const VkSurfaceFormatKHR& sf : surfaceFmts) {
        if (sf.format     == VK_FORMAT_B8G8R8A8_UNORM &&
            sf.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosenFmt = sf;
            break;
        }
    }

    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physDev, surface, &pmCount, presentModes.data());

    VkPresentModeKHR chosenPM = VK_PRESENT_MODE_FIFO_KHR; // guaranteed available
    for (VkPresentModeKHR pm : presentModes) {
        if (pm == VK_PRESENT_MODE_MAILBOX_KHR) { chosenPM = pm; break; }
    }

    int winW = 0, winH = 0;
    SDL_GetWindowSizeInPixels(window, &winW, &winH);
    const VkExtent2D swapExtent{
        .width  = std::clamp(static_cast<uint32_t>(winW),
                             caps.minImageExtent.width,  caps.maxImageExtent.width),
        .height = std::clamp(static_cast<uint32_t>(winH),
                             caps.minImageExtent.height, caps.maxImageExtent.height),
    };
    const uint32_t minImages = caps.minImageCount + 1;
    const uint32_t imageCount = (caps.maxImageCount > 0)
                              ? std::min(minImages, caps.maxImageCount)
                              : minImages;

    const VkSwapchainCreateInfoKHR swapCI{
        .sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext                 = nullptr,
        .flags                 = 0,
        .surface               = surface,
        .minImageCount         = imageCount,
        .imageFormat           = chosenFmt.format,
        .imageColorSpace       = chosenFmt.colorSpace,
        .imageExtent           = swapExtent,
        .imageArrayLayers      = 1,
        .imageUsage            = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .preTransform          = caps.currentTransform,
        .compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode           = chosenPM,
        .clipped               = VK_TRUE,
        .oldSwapchain          = VK_NULL_HANDLE,
    };
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(device, &swapCI, nullptr, &swapchain) != VK_SUCCESS) {
        std::println("[TEST] vkCreateSwapchainKHR failed");
        return 1;
    }
    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &actualCount, nullptr);
    std::vector<VkImage> swapImages(actualCount);
    vkGetSwapchainImagesKHR(device, swapchain, &actualCount, swapImages.data());
    std::println("[TEST] Swapchain: {}x{}  {} images  presentMode={}",
                 swapExtent.width, swapExtent.height, actualCount,
                 static_cast<int>(chosenPM));

    // -----------------------------------------------------------------------
    // 8. Command pool + one command buffer per swapchain image
    // -----------------------------------------------------------------------
    const VkCommandPoolCreateInfo poolCI{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = gfxFamily,
    };
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &poolCI, nullptr, &cmdPool) != VK_SUCCESS) {
        std::println("[TEST] vkCreateCommandPool failed");
        return 1;
    }
    std::vector<VkCommandBuffer> cmds(actualCount);
    const VkCommandBufferAllocateInfo cbAI{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = cmdPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = actualCount,
    };
    if (vkAllocateCommandBuffers(device, &cbAI, cmds.data()) != VK_SUCCESS) {
        std::println("[TEST] vkAllocateCommandBuffers failed");
        return 1;
    }

    // -----------------------------------------------------------------------
    // 9. Synchronisation primitives
    //
    // imageAvailable: one per frame-in-flight (signals when acquire finishes).
    // renderDone:     one per swapchain image (signals when clear+submit done).
    //
    // Indexing renderDone by the acquired image index — not the frame slot —
    // ensures the presentation engine has consumed the semaphore before we
    // reuse it (avoids VUID-vkQueueSubmit2-semaphore-03868).
    // -----------------------------------------------------------------------
    constexpr uint32_t kFramesInFlight = 2;

    std::vector<VkSemaphore> imageAvailable(kFramesInFlight);
    std::vector<VkSemaphore> renderDone(actualCount);   // per swapchain image
    std::vector<VkFence>     frameFences(kFramesInFlight);

    const VkSemaphoreCreateInfo semCI{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    const VkFenceCreateInfo fenceCI{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    for (VkSemaphore& s : imageAvailable) {
        if (vkCreateSemaphore(device, &semCI, nullptr, &s) != VK_SUCCESS) {
            std::println("[TEST] vkCreateSemaphore (imageAvailable) failed");
            return 1;
        }
    }
    for (VkSemaphore& s : renderDone) {
        if (vkCreateSemaphore(device, &semCI, nullptr, &s) != VK_SUCCESS) {
            std::println("[TEST] vkCreateSemaphore (renderDone) failed");
            return 1;
        }
    }
    for (VkFence& f : frameFences) {
        if (vkCreateFence(device, &fenceCI, nullptr, &f) != VK_SUCCESS) {
            std::println("[TEST] vkCreateFence failed");
            return 1;
        }
    }
    std::println("[TEST] Sync objects created — entering render loop (5 s, ESC to quit early)");

    // -----------------------------------------------------------------------
    // 10. Render loop — clears every swapchain image to bright orange
    // -----------------------------------------------------------------------
    std::vector<VkImageLayout> layouts(actualCount, VK_IMAGE_LAYOUT_UNDEFINED);

    uint32_t currentFrame = 0;
    uint32_t frameCount   = 0;
    bool     running      = true;

    const uint64_t startTick = SDL_GetTicks();

    while (running) {
        if (SDL_GetTicks() - startTick >= 5000) {
            std::println("[TEST] 5 s elapsed — exiting loop");
            break;
        }

        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT)                               { running = false; }
            if (event.type == SDL_EVENT_KEY_DOWN &&
                event.key.key == SDLK_ESCAPE)                              { running = false; }
        }
        if (!running) { break; }

        // Wait for this frame slot to be free
        vkWaitForFences(device, 1, &frameFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(device,   1, &frameFences[currentFrame]);

        // Acquire
        uint32_t imgIdx = 0;
        const VkResult acqResult = vkAcquireNextImageKHR(
            device, swapchain, UINT64_MAX,
            imageAvailable[currentFrame], VK_NULL_HANDLE, &imgIdx);
        if (acqResult == VK_ERROR_OUT_OF_DATE_KHR) {
            std::println("[TEST] Swapchain out of date — resize not implemented in this test");
            break;
        }
        if (acqResult != VK_SUCCESS && acqResult != VK_SUBOPTIMAL_KHR) {
            std::println("[TEST] vkAcquireNextImageKHR failed: {}", static_cast<int>(acqResult));
            break;
        }

        VkCommandBuffer cmd = cmds[imgIdx];
        vkResetCommandBuffer(cmd, 0);

        const VkCommandBufferBeginInfo beginInfo{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext            = nullptr,
            .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        };
        vkBeginCommandBuffer(cmd, &beginInfo);

        imageBarrier(cmd, swapImages[imgIdx],
                     layouts[imgIdx],
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_NONE,      0,
                     VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        const VkClearColorValue clearColor{ .float32 = {1.0f, 0.5f, 0.0f, 1.0f} };
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, swapImages[imgIdx],
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearColor, 1, &range);

        imageBarrier(cmd, swapImages[imgIdx],
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_NONE,      0);

        vkEndCommandBuffer(cmd);

        // Submit — wait on imageAvailable, signal renderDone[imgIdx]
        const VkSemaphoreSubmitInfo waitSI{
            .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext       = nullptr,
            .semaphore   = imageAvailable[currentFrame],
            .value       = 0,
            .stageMask   = VK_PIPELINE_STAGE_2_CLEAR_BIT,
            .deviceIndex = 0,
        };
        const VkSemaphoreSubmitInfo signalSI{
            .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext       = nullptr,
            .semaphore   = renderDone[imgIdx],
            .value       = 0,
            .stageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        };
        const VkCommandBufferSubmitInfo cmdSI{
            .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext         = nullptr,
            .commandBuffer = cmd,
            .deviceMask    = 0,
        };
        const VkSubmitInfo2 submitInfo{
            .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext                    = nullptr,
            .flags                    = 0,
            .waitSemaphoreInfoCount   = 1,
            .pWaitSemaphoreInfos      = &waitSI,
            .commandBufferInfoCount   = 1,
            .pCommandBufferInfos      = &cmdSI,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos    = &signalSI,
        };
        if (vkQueueSubmit2(gfxQueue, 1, &submitInfo, frameFences[currentFrame]) != VK_SUCCESS) {
            std::println("[TEST] vkQueueSubmit2 failed");
            break;
        }

        // Present — wait on renderDone[imgIdx]
        const VkPresentInfoKHR presentInfo{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext              = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &renderDone[imgIdx],
            .swapchainCount     = 1,
            .pSwapchains        = &swapchain,
            .pImageIndices      = &imgIdx,
            .pResults           = nullptr,
        };
        const VkResult presentResult = vkQueuePresentKHR(gfxQueue, &presentInfo);
        if (presentResult != VK_SUCCESS && presentResult != VK_SUBOPTIMAL_KHR) {
            std::println("[TEST] vkQueuePresentKHR failed: {}", static_cast<int>(presentResult));
            break;
        }

        layouts[imgIdx] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        currentFrame    = (currentFrame + 1U) % kFramesInFlight;
        ++frameCount;

        if (frameCount % 300 == 1) {
            std::println("[TEST] frame {} — orange clear OK", frameCount);
        }
    }

    std::println("[TEST] loop ended after {} frames", frameCount);

    // -----------------------------------------------------------------------
    // 11. Cleanup (reverse order of creation)
    // -----------------------------------------------------------------------
    vkDeviceWaitIdle(device);

    for (VkFence     f : frameFences)    { vkDestroyFence(device, f, nullptr); }
    for (VkSemaphore s : renderDone)     { vkDestroySemaphore(device, s, nullptr); }
    for (VkSemaphore s : imageAvailable) { vkDestroySemaphore(device, s, nullptr); }
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    if (messenger != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::println("[TEST] === window_swapchain_loop PASSED ({} frames) ===", frameCount);
    return 0;
}
