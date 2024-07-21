cmake_minimum_required(VERSION 3.5)
project(mnist)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH /home/fano/libtorch)

find_package(Torch REQUIRED)

add_executable(mnist main.cpp)
target_compile_features(mnist PUBLIC cxx_range_for)
target_link_libraries(mnist ${TORCH_LIBRARIES})