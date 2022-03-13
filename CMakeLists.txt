cmake_minimum_required(VERSION 3.10)
project(cuda-speedtest CUDA)

enable_language(CXX CUDA)
find_package(CUDA)

file(GLOB_RECURSE MAIN_LIB_SRC *.cu)
add_library(MAIN_LIB ${MAIN_LIB_SRC})

add_executable(main gemm-int8/main.cu)
target_link_libraries(main ${MAIN_LIB} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})