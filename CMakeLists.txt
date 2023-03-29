cmake_minimum_required(VERSION 3.19)
project(Cuda_samples CXX CUDA)
set(CMAKE_CUDA_STANDARD 11)

set(CUDA_DIR "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8")

# CUDA
# 关闭使用静态Runtime
# 如果不加这句的话，编译时会静态链接CUDA
# 如果没有CUDA静态库的话会报错，加上后则为动态链接CUDA
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# 在系统寻找CUDA包路径
# find_package(CUDA REQUIRED)
# 标志文件路径
# include_directories(${CUDA_INCLUDE_DIRS})
# # 包含库文件路径
include_directories(${CUDA_DIR}\\include)
# # 链接库文件
link_directories(${CUDA_DIR}\\lib)
# arch、code根据实际显卡算力填写
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_86;code=sm_86)
# 允许CUDA语法
enable_language(CUDA)

add_subdirectory(Common)
add_subdirectory(0_Introduction)

add_executable(Cuda_samples main.cu)
set_target_properties(
        Cuda_samples
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Cuda_samples PRIVATE common)
target_link_libraries(Cuda_samples PRIVATE introduction)