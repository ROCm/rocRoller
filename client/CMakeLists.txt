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

find_package(BLAS REQUIRED)

<<<<<<< HEAD
target_compile_options(rocroller_gemm PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(rocroller_gemm PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
    rocroller_gemm
    PUBLIC
    rocroller_interface
    common_test_utilities
    CLI11::CLI11
    msgpackc-cxx
)
if(BUILD_SHARED_LIBS)
    target_link_libraries(
        rocroller_gemm PUBLIC ${llvm_yaml_libs}
    )
=======
find_package(CLI11)
if(NOT CLI11_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            CLI11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(CLI11)
    else()
        message(FATAL_ERROR "Failed to find CLI11")
    endif()
>>>>>>> Move next-cmake files to existing build tree
endif()

add_executable(rocroller-gemm
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GEMMParameters.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GraphInspector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/visualize.cpp"
)
target_sources(rocroller-gemm
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/BenchmarkSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/DataParallelGEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GEMMParameters.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GraphInspector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/GraphInspector_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/StreamKGEMMSolution.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/client/visualize.hpp"
)
target_link_libraries(rocroller-gemm
    PRIVATE
        roc::rocroller-interface
        rocroller::common-test-utilities
        CLI11::CLI11
        ${BLAS_LIBRARIES}
)
target_include_directories(rocroller-gemm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(ROCROLLER_BUILD_SHARED_LIBS) # why?
    target_link_libraries(rocroller-gemm PUBLIC ${llvm_yaml_libs})
endif()

add_executable(rocroller-codegen-stress "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen_stress_test.cpp")
target_link_libraries(rocroller-codegen-stress
    PUBLIC
        roc::rocroller-interface
        rocroller::common-test-utilities
        CLI11::CLI11
        ${BLAS_LIBRARIES}
)
if(ROCROLLER_BUILD_SHARED_LIBS) # why are we using this logic
    target_link_libraries(rocroller-codegen-stress PUBLIC ${llvm_yaml_libs})
endif()
