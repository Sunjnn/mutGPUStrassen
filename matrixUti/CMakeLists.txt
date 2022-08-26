add_library(matrixUti matrixUti.cxx blockMatrix.cxx)

set(CMAKE_CUDA_FLAGS "-G -g -Xcompiler -fopenmp -g")