include(GoogleTest)

# ---------------------------------------------------------------------------
# Unit tests — pure CPU, no Vulkan, no GPU resources
#
# Only Hyperion-specific tests live here. Shared CPU tests (math, color space,
# tonemapping, procedural geometry, material library) moved to Harmonia/Aether.
#   - BSDF: GGX/Smith importance-sampling and energy-conservation properties
#           the path tracer relies on.
# ---------------------------------------------------------------------------
set(HYPERION_UNIT_TEST_SOURCES
    unit/test_bsdf.cpp
    unit/test_sampling_plan.cpp
)

add_executable(hyperion_unit_tests ${HYPERION_UNIT_TEST_SOURCES})
target_link_libraries(hyperion_unit_tests
    PRIVATE
        hyperion_lib
        GTest::gtest_main
)
target_include_directories(hyperion_unit_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)
gtest_discover_tests(hyperion_unit_tests PROPERTIES RUN_SERIAL TRUE)

# ---------------------------------------------------------------------------
# Module tests — multi-object GPU workflows (Scene build, Pipeline + SBT)
#
# Catch regressions that span multiple components:
#   - SceneBuild: BLAS + TLAS construction for mesh-only, sphere-only, mixed
#   - SBT:        raygen size == stride (VUID-vkCmdTraceRaysKHR-size-04023),
#                 miss/hit region device addresses, pipeline handle validity
#
# The shared Vulkan test fixture is owned by Harmonia; include it from the
# Harmonia source tree populated by FetchContent.
# ---------------------------------------------------------------------------
set(HYPERION_MODULE_TEST_SOURCES
    module/main.cpp
    module/test_scene_build.cpp
    module/test_sbt.cpp
    module/test_frame_sampling.cpp
    module/test_sms_scene.cpp
    module/test_sms_frame.cpp
)

add_executable(hyperion_module_tests ${HYPERION_MODULE_TEST_SOURCES})
add_dependencies(hyperion_module_tests hyperion_shaders)
set(COMPILE_SLANG_SHADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(COMPILE_SLANG_EXTRA_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/shaders" "${HARMONIA_SHADER_SOURCE_DIR}")
set(COMPILE_SLANG_EXTRA_DEPENDS "${PROJECT_SOURCE_DIR}/shaders/sms_scene.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/sms_transport.slang" "${HARMONIA_SHADER_SOURCE_DIR}/sms_solver.slang"
    "${HARMONIA_SHADER_SOURCE_DIR}/sms_geometry.slang")
compile_slang_shaders(hyperion_test_shaders "${CMAKE_CURRENT_BINARY_DIR}/shaders" sms_scene_test.slang)
unset(COMPILE_SLANG_SHADER_ROOT)
unset(COMPILE_SLANG_EXTRA_INCLUDE_DIRS)
unset(COMPILE_SLANG_EXTRA_DEPENDS)
add_dependencies(hyperion_module_tests hyperion_test_shaders)
target_compile_definitions(hyperion_module_tests PRIVATE HYPERION_TEST_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders")
target_link_libraries(hyperion_module_tests
    PRIVATE
        hyperion_demo
        GTest::gtest
)
target_include_directories(hyperion_module_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${harmonia_SOURCE_DIR}/tests"
)
gtest_discover_tests(hyperion_module_tests PROPERTIES RUN_SERIAL TRUE)

# ---------------------------------------------------------------------------
# System / Integration tests — full end-to-end path tracer render
#
# Creates Context + full pipeline, renders a scene, reads back pixels and
# asserts the image is non-black.  These are the most expensive tests and
# only run at the integration layer.
# ---------------------------------------------------------------------------
set(HYPERION_INTEGRATION_TEST_SOURCES
    integration/test_smoke.cpp
    integration/test_render.cpp
    integration/test_capture_sampling_contract.cpp
)

add_executable(hyperion_integration_tests
    ${HYPERION_INTEGRATION_TEST_SOURCES}
    integration/main.cpp
)
add_dependencies(hyperion_integration_tests hyperion_shaders)
target_link_libraries(hyperion_integration_tests
    PRIVATE
        hyperion_demo
        GTest::gtest
)
target_include_directories(hyperion_integration_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
)
gtest_discover_tests(hyperion_integration_tests PROPERTIES RUN_SERIAL TRUE)
