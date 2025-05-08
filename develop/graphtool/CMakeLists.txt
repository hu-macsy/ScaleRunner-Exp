cmake_minimum_required(VERSION 3.13)

project(graphtool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")

include_directories(${CMAKE_SOURCE_DIR})

set(graphtool_app
    graphtool.cpp
)

add_executable(graphtool ${graphtool_app})

# Debugging Libraries
# target_link_libraries(graphtool PRIVATE asan)

find_package(OpenMP)
target_link_libraries(graphtool PRIVATE OpenMP::OpenMP_CXX)

# Other Libraries
find_package(gdsb REQUIRED)
target_link_libraries(graphtool PRIVATE gdsb::gdsb)

# CLI11
find_package(CLI11)
target_link_libraries(graphtool PRIVATE CLI11::CLI11)

install(TARGETS graphtool DESTINATION bin)
