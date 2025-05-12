cmake_minimum_required(VERSION 3.13)

project(kk_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")

include_directories(${CMAKE_SOURCE_DIR})

set(benchmark_app
    main.cpp
    helpers.cpp
)

add_executable(kk_benchmark ${benchmark_app})

# Debugging Libraries
# target_link_libraries(kk_benchmark PRIVATE asan)

# Concurrency Libraries
find_package(MPI REQUIRED)
target_link_libraries(kk_benchmark PRIVATE MPI::MPI_CXX)

find_package(OpenMP)
target_link_libraries(kk_benchmark PRIVATE OpenMP::OpenMP_CXX)

# Other Libraries
find_package(gdsb REQUIRED)
target_link_libraries(kk_benchmark PRIVATE gdsb::gdsb)

find_package(kklib REQUIRED)
target_link_libraries(kk_benchmark PRIVATE kklib::kklib)

# BEGIN: Libraries required for kklib
# DHB
find_package(dhb REQUIRED)
target_link_libraries(kk_benchmark PRIVATE dhb::dhb)

# CLI11
find_package(CLI11)
target_link_libraries(kk_benchmark PRIVATE CLI11::CLI11)
# END: Libraries required for kklib

install(TARGETS kk_benchmark DESTINATION bin)
