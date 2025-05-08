cmake_minimum_required(VERSION 3.11.4)

project(graphanalysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake.modules/")

include_directories(${CMAKE_SOURCE_DIR})

add_executable(graphanalysis
  main.cpp
)

# needed for debugging
# target_link_libraries(graphanalysis PRIVATE asan)

# CLI11
find_package(CLI11)
target_link_libraries(graphanalysis PRIVATE CLI11::CLI11)

# gdsb
find_package(gdsb REQUIRED)
target_link_libraries(graphanalysis PRIVATE gdsb::gdsb)

# OpenMP
find_package(OpenMP)
target_link_libraries(graphanalysis PRIVATE OpenMP::OpenMP_CXX stdc++fs)

# dhb
find_package(dhb REQUIRED)
target_link_libraries(graphanalysis PRIVATE dhb::dhb)

install(
  TARGETS graphanalysis
  RUNTIME DESTINATION bin
)
