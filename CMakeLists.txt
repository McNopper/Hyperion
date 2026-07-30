cmake_minimum_required(VERSION 3.28)

project(hyperion VERSION 0.7.3 LANGUAGES C CXX)

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

# ============================================================
# AI-first static analysis (clang-tidy / cppcheck / clang-format)
#
# Canonical setup shared by every repo in the pipeline, ported from the
# Hephaestus cpp/ template. Analysis runs only when the generator emits a
# compile database (Ninja/Makefiles) and the compiler is Clang/GNU-
# compatible; otherwise it is skipped and the project configures/builds
# normally. The analysis *targets* are opt-in; in-build analysis is OFF by
# default so the build is never blocked by findings.
#
# Guarded on PROJECT_IS_TOP_LEVEL so the targets are only defined when this
# project is built standalone: several of these repos are consumed via
# FetchContent by downstream pipeline repos, where redefining the targets
# would clash with the consumer's own.
# ============================================================
if(PROJECT_IS_TOP_LEVEL)
    set(PROJECT_REPORTS_DIR ${CMAKE_BINARY_DIR}/reports)
    file(MAKE_DIRECTORY ${PROJECT_REPORTS_DIR})

    set(PROJECT_ANALYSIS_AVAILABLE FALSE)
    if(CMAKE_EXPORT_COMPILE_COMMANDS
       AND (CMAKE_GENERATOR MATCHES "Ninja" OR CMAKE_GENERATOR MATCHES "Makefiles")
       AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
        set(PROJECT_ANALYSIS_AVAILABLE TRUE)
        message(STATUS "AI-first static analysis enabled (${CMAKE_GENERATOR}, ${CMAKE_CXX_COMPILER_ID})")
    else()
        message(STATUS "AI-first static analysis skipped for this toolchain "
            "(${CMAKE_GENERATOR}, ${CMAKE_CXX_COMPILER_ID}); "
            "use Ninja/Makefiles with a Clang- or GNU-compatible compiler to enable it")
    endif()

    file(GENERATE
        OUTPUT ${PROJECT_REPORTS_DIR}/analysis-status.txt
        CONTENT "analysis=$<BOOL:${PROJECT_ANALYSIS_AVAILABLE}>\ngenerator=${CMAKE_GENERATOR}\ncompiler=${CMAKE_CXX_COMPILER_ID}\n"
    )

    option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" ON)
    option(ENABLE_CLANG_TIDY_IN_BUILD "Run clang-tidy during compilation" OFF)
    option(ENABLE_CLANG_FORMAT "Enable clang-format targets" ON)
    option(ENABLE_CPPCHECK "Enable cppcheck static analysis target" ON)
    option(ENABLE_CPPCHECK_IN_BUILD "Run cppcheck alongside the compiler" OFF)

    # All first-party C/C++ sources (clang-format); compiled translation units
    # only (clang-tidy) — headers are analysed transitively via HeaderFilterRegex.
    file(GLOB_RECURSE PROJECT_CXX_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hxx"
    )
    file(GLOB_RECURSE PROJECT_CXX_TUS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
    )

    # ----- clang-tidy -----
    if(ENABLE_CLANG_TIDY AND PROJECT_ANALYSIS_AVAILABLE)
        find_program(CLANG_TIDY_EXE NAMES clang-tidy)
        if(CLANG_TIDY_EXE)
            if(ENABLE_CLANG_TIDY_IN_BUILD)
                set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
                message(STATUS "clang-tidy enabled during build: ${CLANG_TIDY_EXE}")
            else()
                message(STATUS "clang-tidy target enabled: ${CLANG_TIDY_EXE}")
            endif()

            set(CLANG_TIDY_REPORT_DIR ${PROJECT_REPORTS_DIR}/clang-tidy)
            file(MAKE_DIRECTORY ${CLANG_TIDY_REPORT_DIR})

            add_custom_target(tidy
                COMMAND ${CLANG_TIDY_EXE}
                    -p ${CMAKE_BINARY_DIR}
                    --export-fixes=${CLANG_TIDY_REPORT_DIR}/fixes.yaml
                    ${PROJECT_CXX_TUS}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Running clang-tidy; fixes -> ${CLANG_TIDY_REPORT_DIR}/fixes.yaml"
                VERBATIM
            )
        else()
            message(STATUS "clang-tidy not found")
        endif()
    endif()

    # ----- cppcheck -----
    if(ENABLE_CPPCHECK AND PROJECT_ANALYSIS_AVAILABLE)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        include(cppcheck)
    endif()

    # ----- clang-format -----
    if(ENABLE_CLANG_FORMAT)
        find_program(CLANG_FORMAT_EXE NAMES clang-format)
        if(CLANG_FORMAT_EXE)
            add_custom_target(format
                COMMAND ${CLANG_FORMAT_EXE}
                    -i
                    --style=file
                    ${PROJECT_CXX_SOURCES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Formatting source files with clang-format"
                VERBATIM
            )

            add_custom_target(format-check
                COMMAND ${CLANG_FORMAT_EXE}
                    --dry-run
                    --Werror
                    --style=file
                    ${PROJECT_CXX_SOURCES}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Checking source formatting with clang-format"
                VERBATIM
            )

            message(STATUS "clang-format enabled: ${CLANG_FORMAT_EXE}")
        else()
            message(STATUS "clang-format not found")
        endif()
    endif()

    # ----- Aggregate verification targets -----
    # verify:      build + test (fast default).
    # verify-full: verify + format-check + tidy + cppcheck (opt-in strict).
    add_custom_target(verify
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG>
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --build-config $<CONFIG>
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Fast verification: build + test"
        VERBATIM
    )

    add_custom_target(verify-full
        COMMENT "Full verification: verify + format-check + tidy + cppcheck"
    )
    add_dependencies(verify-full verify)
    if(TARGET format-check)
        add_dependencies(verify-full format-check)
    endif()
    if(TARGET cppcheck)
        add_dependencies(verify-full cppcheck)
    endif()
    if(TARGET tidy)
        add_dependencies(verify-full tidy)
    endif()
endif()

# ── Shared libraries: Aether (file format) + Harmonia (common Vulkan) ─────────
# Consumed via FetchContent, pinned to published release tags.  For local
# development against working copies, override FETCHCONTENT_SOURCE_DIR_AETHER /
# FETCHCONTENT_SOURCE_DIR_HARMONIA.
# Aether is declared first so Harmonia reuses this same declaration (dedup by
# name). Aether and Harmonia are pinned to the shared release tag v0.7.2.
FetchContent_Declare(
    aether
    GIT_REPOSITORY https://github.com/McNopper/Aether.git
    GIT_TAG v0.7.2
)
FetchContent_Declare(
    harmonia
    GIT_REPOSITORY https://github.com/McNopper/Harmonia.git
    GIT_TAG v0.7.3
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
        target_compile_options(gtest PRIVATE -Wno-character-conversion)
    endif()
    if(TARGET gtest_main)
        set_property(TARGET gtest_main PROPERTY SYSTEM ON)
        target_compile_options(gtest_main PRIVATE -Wno-character-conversion)
    endif()
    if(TARGET gmock)
        target_compile_options(gmock PRIVATE -Wno-character-conversion)
    endif()
    if(TARGET gmock_main)
        target_compile_options(gmock_main PRIVATE -Wno-character-conversion)
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

# Position-fetch variant of the closest-hit shader for GPUs that support
# VK_KHR_ray_tracing_position_fetch.  closesthit_pf.slang #includes
# closesthit.slang with POSITION_FETCH_ENABLED defined.
set(COMPILE_SLANG_EXTRA_DEFINES "-DPOSITION_FETCH_ENABLED")
set(COMPILE_SLANG_EXTRA_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/closesthit.slang")
compile_slang_shaders(hyperion_shaders_pf "${HYPERION_SHADER_OUTPUT_DIR}" closesthit_pf.slang)
unset(COMPILE_SLANG_EXTRA_DEFINES)
unset(COMPILE_SLANG_EXTRA_DEPENDS)
unset(COMPILE_SLANG_EXTRA_INCLUDE_DIRS)

add_library(hyperion_lib STATIC ${HYPERION_LIBRARY_SOURCES})
add_dependencies(hyperion_lib hyperion_shaders hyperion_shaders_pf)

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
add_dependencies(hyperion hyperion_shaders hyperion_shaders_pf)
target_link_libraries(hyperion PRIVATE hyperion_demo)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Standalone diagnostic executables — no GTest, no Hyperion libs.
# Only SDL3 + Vulkan + volk. Used to isolate driver/presentation issues.
add_executable(hyperion_diag tests/diag/test_sdl_vulkan_init.cpp)
target_link_libraries(hyperion_diag PRIVATE volk hyperion_vulkan_sdk hyperion_project_options)

