include(GoogleTest)

# ---------------------------------------------------------------------------
# Unit tests — pure CPU, no Vulkan, no GPU resources
# ---------------------------------------------------------------------------
set(HYPERION_UNIT_TEST_SOURCES
    unit/test_math.cpp
    unit/test_bsdf.cpp
    unit/test_colorspace.cpp
    unit/test_tonemap.cpp
    unit/test_procedural_geometry.cpp
    unit/test_obj_importer.cpp
    unit/test_material_library.cpp
)

add_executable(hyperion_unit_tests ${HYPERION_UNIT_TEST_SOURCES})
target_link_libraries(hyperion_unit_tests
    PRIVATE
        hyperion_lib
        GTest::gtest_main
)
target_include_directories(hyperion_unit_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)
gtest_discover_tests(hyperion_unit_tests)

# ---------------------------------------------------------------------------
# Component tests — single Vulkan object lifecycle (Buffer, Image, CommandPool)
#
# Catch regressions at the lowest GPU level:
#   - Buffer: vmaUnmapMemory bug, host-mapped round-trip, staging upload,
#             device address
#   - Image:  creation, layout transition, pixel readback via clear + copy
#   - CommandPool: one-shot command buffer submission, vkCmdFillBuffer verify
# ---------------------------------------------------------------------------
set(HYPERION_COMPONENT_TEST_SOURCES
    component/main.cpp
    component/test_buffer.cpp
    component/test_image.cpp
    component/test_commandpool.cpp
)

add_executable(hyperion_component_tests ${HYPERION_COMPONENT_TEST_SOURCES})
target_link_libraries(hyperion_component_tests
    PRIVATE
        hyperion_demo
        GTest::gtest
)
target_include_directories(hyperion_component_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
gtest_discover_tests(hyperion_component_tests)

# ---------------------------------------------------------------------------
# Module tests — multi-object GPU workflows (Scene build, Pipeline + SBT)
#
# Catch regressions that span multiple components:
#   - SceneBuild: BLAS + TLAS construction for mesh-only, sphere-only, mixed
#   - SBT:        raygen size == stride (VUID-vkCmdTraceRaysKHR-size-04023),
#                 miss/hit region device addresses, pipeline handle validity
# ---------------------------------------------------------------------------
set(HYPERION_MODULE_TEST_SOURCES
    module/main.cpp
    module/test_scene_build.cpp
    module/test_sbt.cpp
)

add_executable(hyperion_module_tests ${HYPERION_MODULE_TEST_SOURCES})
add_dependencies(hyperion_module_tests hyperion_shaders)
target_link_libraries(hyperion_module_tests
    PRIVATE
        hyperion_demo
        GTest::gtest
)
target_include_directories(hyperion_module_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
gtest_discover_tests(hyperion_module_tests)

# ---------------------------------------------------------------------------
# System / Integration tests — full end-to-end path tracer render
#
# Creates Context + full pipeline, renders a scene, reads back pixels and
# asserts the image is non-black.  These are the most expensive tests and
# only run at the integration layer.
# ---------------------------------------------------------------------------
set(HYPERION_INTEGRATION_TEST_SOURCES
    integration/test_smoke.cpp
    integration/test_context.cpp
    integration/test_render.cpp
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
gtest_discover_tests(hyperion_integration_tests)
