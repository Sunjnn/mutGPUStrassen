add_library(threadPool threadPool.cu)

target_include_directories(threadPool
    PUBLIC "${PROJECT_SOURCE_DIR}/matrixUti" "${PROJECT_SOURCE_DIR}/cudaUti")

target_link_libraries(threadPool
    PUBLIC matrixUti cudaUti)

set(CMAKE_CUDA_FLAGS "-G -g -Xcompiler -fopenmp -g")
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
# find_package(cuBLAS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
