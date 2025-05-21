################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

#find_package(isa_spec_manager)
#if(NOT isa_spec_manager_FOUND)
#    if(ROCROLLER_ENABLE_CLIENT)
#        FetchContent_Declare(
#            isa_spec_manager
#            GIT_REPOSITORY https://github.com/GPUOpen-Tools/isa_spec_manager.git
#            GIT_TAG 1c6fcc8072c7a94ceedb734cf06fd8a08487c4b3
#        )
#        FetchContent_MakeAvailable(isa_spec_manager)
#    else()
#        message(FATAL_ERROR "Failed to find isa_spec_manager")
#    endif()
#endif()

# we build all of isa_spec_manager if we don't do the following.
# fetch content is not the correct tool
#set_property(DIRECTORY ${isa_spec_manager_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL ON)
#add_library(
#    isa_spec_manager OBJECT
#)

#target_include_directories(
#    isa_spec_manager SYSTEM
#    PUBLIC ${isa_spec_manager_SOURCE_DIR}/include
#           ${isa_spec_manager_SOURCE_DIR}/source/common
#           ${isa_spec_manager_SOURCE_DIR}/source/third_party/tinyxml2
#)

if(ROCROLLER_ENABLE_FETCH)
    FetchContent_Declare(
        mrisa_xml
        URL https://gpuopen.com/download/machine-readable-isa/AMD_GPU_MR_ISA_XML-2024_08_22.zip
    )
    FetchContent_MakeAvailable(mrisa_xml)
endif()

add_executable(GPUArchitectureGenerator
    "${ROCROLLER_GPUARCH_DIR}/source/GPUArchitectureGenerator.cpp"
    "${ROCROLLER_LIB_DIR}/source/GPUArchitecture.cpp"
    "${ROCROLLER_LIB_DIR}/source/GPUArchitectureLibrary.cpp"
    "${ROCROLLER_LIB_DIR}/source/Assemblers/Assembler.cpp"
    "${ROCROLLER_LIB_DIR}/source/Costs/Cost.cpp"
    "${ROCROLLER_LIB_DIR}/source/DataTypes_F6_Utils.cpp"
    "${ROCROLLER_LIB_DIR}/source/DataTypes_F8_Utils.cpp"
    "${ROCROLLER_LIB_DIR}/source/DataTypes_BF16_Utils.cpp"
    "${ROCROLLER_LIB_DIR}/source/Error.cpp"
    "${ROCROLLER_LIB_DIR}/source/Logging.cpp"
    "${ROCROLLER_LIB_DIR}/source/Scheduler.cpp"
    "${ROCROLLER_LIB_DIR}/source/Utilities/Settings.cpp"
    "${ROCROLLER_LIB_DIR}/source/Timer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/isa_xml_reader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tinyxml2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amdisa_utility.cpp"
)

target_include_directories(GPUArchitectureGenerator PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(ROCROLLER_ENABLE_LLVM)
    target_compile_definitions(GPUArchitectureGenerator PRIVATE ROCROLLER_USE_LLVM)
    target_link_libraries(GPUArchitectureGenerator PRIVATE ${llvm_yaml_libs})
else()
    target_compile_definitions(GPUArchitectureGenerator PRIVATE ROCROLLER_USE_YAML_CPP)
    target_link_libraries(GPUArchitectureGenerator PRIVATE yaml-cpp::yaml-cpp)
endif()

target_include_directories(GPUArchitectureGenerator
    PRIVATE 
        "${ROCROLLER_LIB_DIR}/../GPUArchitectureGenerator/include"
        "${ROCROLLER_LIB_DIR}/include"
)
target_compile_features(GPUArchitectureGenerator PRIVATE cxx_std_20)
target_link_libraries(GPUArchitectureGenerator
    PRIVATE
        ${msgpack_libs}
        ${llvm_libs}
        spdlog::spdlog_header_only
        hip::host
        fmt::fmt-header-only
)
target_compile_definitions(GPUArchitectureGenerator
    PRIVATE
       ROCROLLER_USE_HIP
       __HIP_PLATFORM_AMD__
)

file(GLOB YAML_FILES "${ROCROLLER_LIB_DIR}/../GPUArchitectureGenerator/pregenerated/*.yaml")
set(YAML_ARGS "")
foreach(YAML_FILE ${YAML_FILES})
    list(APPEND YAML_ARGS --yaml "${YAML_FILE}")
endforeach()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml
    COMMAND
        $<TARGET_FILE:GPUArchitectureGenerator>
        "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
        ${CMAKE_CXX_COMPILER}
        -Y ${YAML_ARGS}
    DEPENDS GPUArchitectureGenerator "${YAML_FILES}"
)

add_custom_command(
    DEPENDS GPUArchitectureGenerator "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack"
    COMMAND
        $<TARGET_FILE:GPUArchitectureGenerator>
        "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack"
        ${CMAKE_CXX_COMPILER}
        --yaml "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml"
)

if(ROCROLLER_EMBED_ARCH_DEF)
    cmrc_add_resource_library(
        GPUArchitecture_def_library
        ALIAS rocroller::GPUArchitecture_def
        NAMESPACE rocRoller
        WHENCE ${CMAKE_CURRENT_BINARY_DIR}
        PREFIX resources
        "${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack"
    )
    set_target_properties(
        GPUArchitecture_def_library PROPERTIES POSITION_INDEPENDENT_CODE ON
    )
endif()

add_custom_target(
    GPUArchitecture_def ALL
    DEPENDS
        GPUArchitecture_def_library
        ${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.yaml
        ${CMAKE_CURRENT_BINARY_DIR}/GPUArchitecture_def.msgpack
)
