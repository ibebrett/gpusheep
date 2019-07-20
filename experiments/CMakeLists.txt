cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

add_executable(test1
  test.cu
)

target_compile_features(test1 PUBLIC cxx_std_11)
