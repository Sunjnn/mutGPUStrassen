cmake_minimum_required(VERSION 3.20)

project(multiGPUStrassen LANGUAGES CUDA CXX)
set(CMAKE_cxx_COMPILE /usr/bin/g++)
set(CMAKE_CUDA_COMPILE /usr/local/cuda-11.7/bin/nvcc)

# add_library(cudaUti cudaUti.cu cudaUti.cuh)
add_subdirectory(cudaUti)
add_subdirectory(matrixUti)
add_subdirectory(threadPool)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(CUDA)
# find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
# set(CMAKE_CUDA_ARCHITECTURES 80)
# set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
find_package(cuBLAS)

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cu)

target_link_libraries(${PROJECT_NAME}
    PUBLIC cudaUti matrixUti threadPool
    PRIVATE ${CUBLAS_LIBRARIES}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC "${PROJECT_SOURCE_DIR}/cudaUti" "${PROJECT_SOURCE_DIR}/matrixUti" "${PROJECT_SOURCE_DIR}/threadPool"
)

set(CMAKE_CUDA_FLAGS "-G -g -Xcompiler -fopenmp")

# target_compile_features(cublasMatMul PRIVATE cxx_std_11)
# target_link_libraries(cublasMatMul PRIVATE ${CUBLAS_LIBRARIES})