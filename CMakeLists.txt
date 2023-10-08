cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(TEST_INCLUDE_FILE LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")

find_package(PythonInterp REQUIRED)

file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
list(APPEND TEST_EXTERNAL_LIB ${TORCH_LIBRARIES})



file(GLOB TEST_SRC
  src/*.cc
  src/*.cu
)

add_library(test SHARED ${TEST_SRC})
target_link_libraries(test ${TEST_EXTERNAL_LIB})
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test PROPERTIES CUDA_STANDARD 14)
set_target_properties(test PROPERTIES POSITION_INDEPENDENT_CODE ON)
