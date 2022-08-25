# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
# find_package(cuBLAS)

set(CMAKE_CUDA_FLAGS "-G -g -Xcompiler -fopenmp")

add_library(cudaUti cudaUti.cu gemmStrassen.cu)

target_include_directories(cudaUti
    PUBLIC "${PROJECT_SOURCE_DIR}/matrixUti")

target_link_libraries(cudaUti
    PUBLIC matrixUti)

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
