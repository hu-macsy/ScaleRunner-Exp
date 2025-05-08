cmake_minimum_required(VERSION 3.13)

project(io_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")

include_directories(${CMAKE_SOURCE_DIR})

set(io_benchmark_app
    io_benchmark_main.cpp
)

add_executable(io_benchmark ${io_benchmark_app})

# Debugging Libraries
# target_link_libraries(io_benchmark PRIVATE asan)

find_package(MPI REQUIRED)
target_link_libraries(io_benchmark PRIVATE MPI::MPI_CXX)

find_package(OpenMP)
target_link_libraries(io_benchmark PRIVATE OpenMP::OpenMP_CXX)

# Other Libraries
find_package(gdsb REQUIRED)
target_link_libraries(io_benchmark PRIVATE gdsb::gdsb)

find_package(dhb REQUIRED)
target_link_libraries(io_benchmark PRIVATE dhb::dhb)

find_package(scalerunner REQUIRED)
target_link_libraries(io_benchmark PRIVATE scalerunner::scalerunner)

# CLI11
find_package(CLI11)
target_link_libraries(io_benchmark PRIVATE CLI11::CLI11)

# TBB
find_package(TBB REQUIRED)
target_link_libraries(io_benchmark PRIVATE TBB::tbb)

install(TARGETS io_benchmark DESTINATION bin)
