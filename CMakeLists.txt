cmake_minimum_required(VERSION 3.28)

project(hyperion VERSION 0.4.0 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(FetchContent)
include(Platform)
include(VulkanSDK)
# compile_slang_shaders() is provided by Harmonia (cmake/CompileShaders.cmake),
# loaded when Harmonia is added via FetchContent below.

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(VK_NO_PROTOTYPES)

# ── Shared libraries: Aether (file format) + Harmonia (common Vulkan) ─────────
# Consumed via FetchContent, pinned to published release tags.  For local
# development against working copies, override FETCHCONTENT_SOURCE_DIR_AETHER /
# FETCHCONTENT_SOURCE_DIR_HARMONIA.
# Aether is declared first so Harmonia reuses this same declaration (dedup by
# name); both pin v0.1.0.
FetchContent_Declare(
    aether
    GIT_REPOSITORY https://github.com/McNopper/Aether.git
    GIT_TAG v0.4.0
)
FetchContent_Declare(
    harmonia
    GIT_REPOSITORY https://github.com/McNopper/Harmonia.git
    GIT_TAG v0.4.0
)
FetchContent_MakeAvailable(aether harmonia)

set(HYPERION_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

set(HYPERION_ENTRY_SHADERS
    raygen.slang
    closesthit.slang
    intersection.slang
    miss.slang
    shadow_miss.slang
)

# Hyperion-specific sources only. Everything common now lives in Harmonia, and
# the file-format parsers in Aether; both are linked via harmonia::harmonia.
set(HYPERION_LIBRARY_SOURCES
    src/hyperion/renderer/PathTracer.cpp
    src/hyperion/renderer/ShaderBindingTable.cpp
    src/hyperion/scene/Scene.cpp
)

set(HYPERION_DEMO_SOURCES
    src/demo/Application.cpp
)

set(HYPERION_APP_SOURCES
    src/demo/main.cpp
)

# Volk is provided by Harmonia (created during FetchContent above); only define
# it ourselves if Harmonia did not (e.g. when building Hyperion in isolation).
if(NOT TARGET volk)
    add_library(volk STATIC "${VULKANSDK_VOLK_SOURCE}")
    target_include_directories(volk SYSTEM PUBLIC "${VULKANSDK_INCLUDE_DIR}")
    set_target_properties(volk PROPERTIES LINKER_LANGUAGE C)
endif()

add_library(hyperion_project_options INTERFACE)
hyperion_configure_target(hyperion_project_options)

add_library(hyperion_vulkan_sdk INTERFACE)
target_include_directories(hyperion_vulkan_sdk SYSTEM INTERFACE "${VULKANSDK_INCLUDE_DIR}")
target_link_libraries(hyperion_vulkan_sdk INTERFACE ${VULKANSDK_LIBRARIES})

if(BUILD_TESTING)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    if(TARGET gtest)
        set_property(TARGET gtest PROPERTY SYSTEM ON)
    endif()
    if(TARGET gtest_main)
        set_property(TARGET gtest_main PROPERTY SYSTEM ON)
    endif()
    if(NOT TARGET GTest::gtest AND TARGET gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(NOT TARGET GTest::gtest_main AND TARGET gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endif()

# Harmonia's shared Slang modules (env_sample, math, bsdf, …) are the source
# of truth; add their directory so slangc resolves imports from there.
set(COMPILE_SLANG_EXTRA_INCLUDE_DIRS "${HARMONIA_SHADER_SOURCE_DIR}")
compile_slang_shaders(hyperion_shaders "${HYPERION_SHADER_OUTPUT_DIR}" ${HYPERION_ENTRY_SHADERS})
unset(COMPILE_SLANG_EXTRA_INCLUDE_DIRS)

add_library(hyperion_lib STATIC ${HYPERION_LIBRARY_SOURCES})
add_dependencies(hyperion_lib hyperion_shaders)

target_include_directories(hyperion_lib
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# harmonia::harmonia brings the common Vulkan library plus Aether, Vulkan SDK,
# volk, stb and OpenEXR (incl. HARMONIA_HAS_OPENEXR) transitively.
target_link_libraries(hyperion_lib
    PUBLIC hyperion_project_options harmonia::harmonia
)

target_compile_definitions(hyperion_lib
    PUBLIC HYPERION_SHADER_DIR=\"${HYPERION_SHADER_OUTPUT_DIR}\"
           HYPERION_ASSETS_DIR=\"${AETHER_ASSETS_DIR}\"
)

add_library(hyperion_demo STATIC ${HYPERION_DEMO_SOURCES})
target_link_libraries(hyperion_demo PUBLIC hyperion_lib)

add_executable(hyperion ${HYPERION_APP_SOURCES})
add_dependencies(hyperion hyperion_shaders)
target_link_libraries(hyperion PRIVATE hyperion_demo)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Standalone diagnostic executables — no GTest, no Hyperion libs.
# Only SDL3 + Vulkan + volk. Used to isolate driver/presentation issues.
add_executable(hyperion_diag tests/diag/test_sdl_vulkan_init.cpp)
target_link_libraries(hyperion_diag PRIVATE volk hyperion_vulkan_sdk hyperion_project_options)
