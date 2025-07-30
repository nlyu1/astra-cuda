cmake_minimum_required(VERSION 4.0.3)
project(astra-cuda LANGUAGES C CXX CUDA)
# Propagate CLI arguments
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS})
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# CUDA and C++ flags 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87 89 90 100 101 120)

# Force CUDA to use the conda compiler if available
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

##### Auto-detect system CUDA installation #####
find_package(CUDAToolkit REQUIRED)
message(STATUS "✅ Using CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
# Set legacy CUDA variables for PyTorch compatibility
set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})
set(CUDA_FOUND TRUE)


##### Find python with dev-support #####
# Else if lib A requires python (but not python dev)
#     while lib B requires python dev, then fetch-including lib A then B 
#     throws python (dev) not found error

# Prioritize conda environment python if CONDA_PREFIX is set
if(DEFINED ENV{CONDA_PREFIX})
    set(Python_ROOT_DIR "$ENV{CONDA_PREFIX}")
    message(STATUS "🐍 Using conda environment: $ENV{CONDA_PREFIX}")
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
message(STATUS "✅ Found Python executable: ${Python_EXECUTABLE}")
include(FetchContent)

##### Interface library bundles dependencies and compiler flags #####
add_library(astra_deps INTERFACE)


# ============================================================================
# LibTorch support. Run in an activated conda environment with pytorch installed. 
# ============================================================================
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE Torch_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "✅ Found Torch_DIR: ${Torch_DIR}")
find_package(Torch REQUIRED HINTS "${Torch_DIR}")
target_link_libraries(astra_deps INTERFACE "${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ============================================================================
# Pybind11 support 
# ============================================================================
set(PYBIND11_FINDPYTHON NEW)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v3.0.0
  SYSTEM
)
FetchContent_MakeAvailable(pybind11) 
target_link_libraries(astra_deps INTERFACE pybind11::embed)

# ============================================================================
# Dependencies: fmt for string formatting 
# ============================================================================
# set(PYBIND11_FINDPYTHON NEW)

# Fetch fmt from GitHub
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
)
FetchContent_MakeAvailable(fmt)
target_link_libraries(astra_deps INTERFACE fmt::fmt)

# ============================================================================
# Target Properties: Compiler Flags, Linker Options, and Clang-Tidy
# ============================================================================
target_compile_features(astra_deps INTERFACE cxx_std_20)

# Add CUDA runtime libraries
# target_link_libraries(astra_deps INTERFACE CUDA::cudart_static CUDA::cuda_driver)

# --- Clang-Tidy (Debug builds only) ---
# This property is inherited by all targets that link to astra_deps.
# The generator expression ensures it's only active for Debug builds.
set(CLANG_TIDY_CHECKS "*")
set_property(TARGET astra_deps APPEND PROPERTY
    INTERFACE_CXX_CLANG_TIDY
        $<$<CONFIG:Debug>:${CLANG_TIDY_EXE};-checks=${CLANG_TIDY_CHECKS};-header-filter=^${CMAKE_SOURCE_DIR}/src/>)

# --- Common Compiler Flags (all build types) ---
target_compile_options(astra_deps INTERFACE
    -fopenmp
    -fPIC
    -Wformat=2
    -Wunused
    -Wno-stringop-overread)
# --- Build-Specific Compiler Flags ---
# Debug: Add debug symbols, disable optimization, and enable sanitizers.
target_compile_options(astra_deps INTERFACE
    $<$<CONFIG:Debug>:
        -g
        -O0
        -fsanitize=address,undefined
        -fno-omit-frame-pointer>)
# Release: Add optimization flags.
target_compile_options(astra_deps INTERFACE
    $<$<CONFIG:Release>:
        -O3
        -mprefer-vector-width=512
        -march=native
        -mtune=native
        -DNDEBUG>)
target_compile_options(astra_deps INTERFACE
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:-O3>>
    $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:-G>>
)
# --- Linker Options ---
target_link_options(astra_deps INTERFACE
    -fopenmp
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>)


# Add utilities: library `astra_utils`
# ============================================================================
add_subdirectory(utils)

# Uniform objects. Later add_subdirectory all use this
set(ASTRA_OBJECTS 
    astra_utils
    astra_deps
)
# Add order-matching functionalities 
add_subdirectory(order_matching)
list(APPEND ASTRA_OBJECTS order_matching)


add_executable(main main.cc)
target_link_libraries(main ${ASTRA_OBJECTS})

# building tests
enable_testing()
add_subdirectory(tests)