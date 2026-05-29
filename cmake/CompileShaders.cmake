include_guard(GLOBAL)

function(compile_slang_shaders target output_dir)
    if(NOT SLANGC_EXECUTABLE)
        message(FATAL_ERROR "SLANGC_EXECUTABLE is not set. Include VulkanSDK.cmake before CompileShaders.cmake.")
    endif()

    if(ARGC LESS 3)
        message(FATAL_ERROR "compile_slang_shaders(target output_dir shader1 shader2 ...) requires at least one shader.")
    endif()

    set(_shader_root "${CMAKE_SOURCE_DIR}/shaders")
    set(_support_shaders
        "${_shader_root}/common.slang"
        "${_shader_root}/math.slang"
        "${_shader_root}/bsdf.slang"
        "${_shader_root}/env.slang"
    )
    set(_outputs)

    foreach(_shader IN LISTS ARGN)
        if(IS_ABSOLUTE "${_shader}")
            set(_input_shader "${_shader}")
        else()
            set(_input_shader "${_shader_root}/${_shader}")
        endif()

        get_filename_component(_shader_name_we "${_input_shader}" NAME_WE)
        set(_output_shader "${output_dir}/${_shader_name_we}.spv")

        add_custom_command(
            OUTPUT "${_output_shader}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
            COMMAND "${SLANGC_EXECUTABLE}" "${_input_shader}" -target spirv -profile spirv_1_6 -g0 -O2 -I "${_shader_root}" -o "${_output_shader}"
            DEPENDS
                "${_input_shader}"
                ${_support_shaders}
            COMMENT "Compiling Slang shader ${_shader_name_we}.slang"
            VERBATIM
            COMMAND_EXPAND_LISTS
        )

        list(APPEND _outputs "${_output_shader}")
    endforeach()

    add_custom_target(${target} DEPENDS ${_outputs})
    set(${target}_OUTPUTS "${_outputs}" PARENT_SCOPE)
endfunction()
