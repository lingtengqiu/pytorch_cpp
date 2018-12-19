cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)
add_executable(launch minist.cpp)
target_link_libraries(launch "${TORCH_LIBRARIES}")
set_property(TARGET launch PROPERTY CXX_STANDARD 11)
