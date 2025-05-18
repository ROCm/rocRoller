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

include(CTest)

if(YAML_BACKEND STREQUAL "YAML_CPP")
    target_compile_definitions(
        rocRollerTests PUBLIC ROCROLLER_TESTS_USE_YAML_CPP
    )
endif()

FetchContent_Declare(
    mxDataGenerator
    GIT_REPOSITORY "https://github.com/ROCm/mxDataGenerator.git"
    GIT_TAG e74b2c1b6f357c053b27958177ca454ba21918b7
)
FetchContent_MakeAvailable(mxDataGenerator)

FetchContent_Declare(
    Catch2    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG devel
)
FetchContent_MakeAvailable(Catch2)

add_executable(rocroller-tests-catch)

#target_compile_options(rocroller-tests-catch PUBLIC ${EXTRA_COMPILE_OPTIONS})
#target_link_options(rocroller-tests-catch PUBLIC ${EXTRA_LINK_OPTIONS})
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(rocroller-tests-catch
    PRIVATE
        common-test-utilities
        Catch2::Catch2WithMain
        BLAS::BLAS
)

#if(YAML_BACKEND STREQUAL "YAML_CPP")
#    target_compile_definitions(rocroller-tests-catch PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
#endif()

add_subdirectory(common)
add_subdirectory(catch)

#include(Catch)
#catch_discover_tests(
#    rocroller-tests-catch
#    REPORTER JUnit
#    OUTPUT_DIR ${TEST_REPORT_DIR}
#    OUTPUT_SUFFIX ".xml"
#    DISCOVERY_MODE PRE_TEST
#    EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
#    TEST_SPEC "[gpu]"
#    PROPERTIES "LABELS"
#               "CATCH"
#               "LABELS"
#               "GPU"
#    SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
#)
#catch_discover_tests(
#    rocroller-tests-catch
#    REPORTER JUnit
#    OUTPUT_DIR ${TEST_REPORT_DIR}
#    OUTPUT_SUFFIX ".xml"
#    DISCOVERY_MODE PRE_TEST
#    EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
#    TEST_SPEC "~[gpu]"
#    PROPERTIES "LABELS" "CATCH"
#    SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
#)
