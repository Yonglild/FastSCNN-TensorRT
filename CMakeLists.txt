cmake_minimum_required(VERSION 3.10)
project(TensorrtxPractice)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

include_directories(./)

# cuda
include_directories(/usr/local/cuda-10.0/include)
link_directories(/usr/local/cuda-10.0/lib64)

# tensorrt
include_directories(/home/wyl/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/include)
link_directories(/home/wyl/TensorRT-7.0.0.11/targets/x86_64-linux-gnu/lib)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS}
        ./fastscnn.h)

add_executable(fastscnn ./fastscnn.cpp)
target_link_libraries(fastscnn nvinfer cudart ${OpenCV_LIBS})