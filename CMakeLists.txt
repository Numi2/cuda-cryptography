cmake_minimum_required(VERSION 3.24)

project(cuda_pqc_bench
  VERSION 0.1.0
  DESCRIPTION "CUDA/C++ benchmarks for proof-system-adjacent cryptographic primitives"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)

option(CPB_ENABLE_CUDA "Build CUDA kernels when a CUDA compiler is available" ON)

if(CPB_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70)
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(CPB_WITH_CUDA ON)
  else()
    message(WARNING "CUDA compiler not found; building CPU-only targets. Pass -DCPB_ENABLE_CUDA=OFF to silence this warning.")
    set(CPB_WITH_CUDA OFF)
  endif()
else()
  set(CPB_WITH_CUDA OFF)
endif()

add_library(cpb_core
  src/merkle_cpu.cpp
  src/ntt_cpu.cpp
  src/sha256.cpp
  src/vector_cpu.cpp)

target_include_directories(cpb_core PUBLIC include)
target_compile_features(cpb_core PUBLIC cxx_std_20)

if(CPB_WITH_CUDA)
  add_library(cpb_cuda src/cuda_primitives.cu)
  target_include_directories(cpb_cuda PUBLIC include)
  target_link_libraries(cpb_cuda PUBLIC cpb_core CUDA::cudart)
  target_compile_definitions(cpb_cuda PUBLIC CPB_WITH_CUDA=1)
  target_compile_options(cpb_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --device-int128>)
  set_target_properties(cpb_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
endif()

add_executable(cpb_correctness_tests tests/correctness_tests.cpp)
target_link_libraries(cpb_correctness_tests PRIVATE cpb_core)

add_executable(cpb_bench bench/bench.cpp)
target_link_libraries(cpb_bench PRIVATE cpb_core)

if(CPB_WITH_CUDA)
  target_link_libraries(cpb_correctness_tests PRIVATE cpb_cuda)
  target_link_libraries(cpb_bench PRIVATE cpb_cuda)
  target_compile_definitions(cpb_correctness_tests PRIVATE CPB_WITH_CUDA=1)
  target_compile_definitions(cpb_bench PRIVATE CPB_WITH_CUDA=1)
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME correctness COMMAND cpb_correctness_tests)
endif()
