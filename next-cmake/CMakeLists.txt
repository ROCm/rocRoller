cmake_minimum_required(VERSION 3.21.0)
project(rocroller VERSION 1.0.0)

set(ROCROLLER_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
set(ROCROLLER_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../test")

include(FetchContent)

set(ROCROLLER_ENABLE_HIP ON CACHE BOOL "")
set(ROCROLLER_ENABLE_CMRC OFF CACHE BOOL "")
set(ROCROLLER_ENABLE_LLVM ON CACHE BOOL "")
set(ROCROLLER_ENABLE_CLIENT OFF CACHE BOOL "")
set(ROCROLLER_BUILD_TESTING ON CACHE BOOL "")

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.x
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    libdivide
    GIT_REPOSITORY https://github.com/ridiculousfish/libdivide.git
    GIT_TAG master
)
FetchContent_MakeAvailable(libdivide)

add_library(rocroller)
add_library(roc::rocroller ALIAS rocroller)

if(ROCROLLER_ENABLE_CMRC)
    FetchContent_Declare(
        cmrc
        GIT_REPOSITORY https://github.com/vector-of-bool/cmrc.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(cmrc)
    target_compile_definitions(rocroller PRIVATE ROCROLLER_EMBED_ARCH_DEF)
endif()


if(ROCROLLER_ENABLE_HIP)
    find_package(hip REQUIRED)
    target_compile_definitions(rocroller 
        PUBLIC 
            ROCROLLER_USE_HIP
            __HIP_PLATFORM_AMD__
    )
    find_package(LLD REQUIRED)
    target_link_libraries(rocroller PRIVATE hip::host)
    target_include_directories(rocroller PRIVATE "${LLD_INCLUDE_DIRS}")
endif()

if(ROCROLLER_ENABLE_LLVM)
    find_package(LLVM REQUIRED)
    target_link_libraries(rocroller PRIVATE LLVMObjectYAML)
    target_compile_definitions(rocroller PRIVATE ROCROLLER_USE_LLVM)
endif()

target_link_libraries(rocroller
    PUBLIC
        fmt::fmt
    PRIVATE
        spdlog::spdlog
        libdivide::libdivide
)

target_include_directories(rocroller 
    PUBLIC
        $<BUILD_INTERFACE:${ROCROLLER_LIB_DIR}/include>
)

target_compile_features(rocroller PUBLIC cxx_std_20)

add_subdirectory(host-library)

if(ROCROLLER_ENABLE_CLIENT)
    add_subdirectory(client)
endif()

if(ROCROLLER_BUILD_TESTING)
    add_subdirectory(test)
endif()
