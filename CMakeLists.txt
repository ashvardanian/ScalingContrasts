cmake_minimum_required(VERSION 3.28)
project(scalingcontrasts LANGUAGES CXX CUDA)

set(SCALINGCONTRASTS_CUDA_ARCHITECTURES "100f-real" CACHE STRING "GPU code to compile, as SASS")
find_package(CUDAToolkit REQUIRED)
find_path(SCALINGCONTRASTS_VERBS_INCLUDE infiniband/verbs.h REQUIRED)
find_library(SCALINGCONTRASTS_VERBS_LIBRARY ibverbs REQUIRED)
# Multi-index `operator[]` needs C++23, which `nvcc` accepts from 13.1, but CMake's `nvcc` dialect table stops at 20.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1 AND NOT CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "-std=c++23")
    set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "-std=c++23")
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
# NumKong links the shared CUDA runtime, and its kernels register only with the runtime that loads them.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(NUMKONG_BUILD_CUDA ON)
set(NUMKONG_CUDA_ARCHITECTURES "${SCALINGCONTRASTS_CUDA_ARCHITECTURES}")
set(NUMKONG_BUILD_SHARED OFF)
set(NUMKONG_INSTALL OFF)
include(FetchContent)
FetchContent_Declare(numkong GIT_REPOSITORY https://github.com/ashvardanian/NumKong
                             GIT_TAG 2dea1586c8c26d1ce9de230b0165078480c63a9a)
FetchContent_MakeAvailable(numkong)

add_executable(scalingcontrasts scalingcontrasts.cu)
target_include_directories(scalingcontrasts PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(scalingcontrasts SYSTEM PRIVATE ${SCALINGCONTRASTS_VERBS_INCLUDE})
target_link_libraries(scalingcontrasts PRIVATE numkong::static CUDA::cuda_driver ${SCALINGCONTRASTS_VERBS_LIBRARY})
target_compile_options(scalingcontrasts PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# SASS rather than PTX, so drivers older than the toolkit run it; Clang links the C++ runtime in statically.
set_target_properties(scalingcontrasts PROPERTIES CUDA_ARCHITECTURES "${SCALINGCONTRASTS_CUDA_ARCHITECTURES}"
                                                  LINKER_LANGUAGE CXX BUILD_RPATH "$ORIGIN")
target_link_options(scalingcontrasts PRIVATE -static-libstdc++ -static-libgcc)
