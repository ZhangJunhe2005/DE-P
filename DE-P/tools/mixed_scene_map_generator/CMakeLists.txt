cmake_minimum_required(VERSION 3.10)
project(yopo_mixed_scene_map_generator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED YOPO_SIMULATOR_SOURCE)
  message(FATAL_ERROR "YOPO_SIMULATOR_SOURCE is required")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(ROSCPP REQUIRED roscpp)
link_directories(${ROSCPP_LIBRARY_DIRS})
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)
find_library(
  YAML_CPP_LIBRARY NAMES yaml-cpp
  PATHS /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH REQUIRED
)

add_executable(
  yopo_mixed_scene_map_generator
  main.cpp
  "${YOPO_SIMULATOR_SOURCE}/src/maps.cpp"
  "${YOPO_SIMULATOR_SOURCE}/src/perlinnoise.cpp"
)
target_include_directories(
  yopo_mixed_scene_map_generator PRIVATE
  "${YOPO_SIMULATOR_SOURCE}/include"
  ${ROSCPP_INCLUDE_DIRS}
  /opt/ros/noetic/include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
target_compile_options(yopo_mixed_scene_map_generator PRIVATE -O3)
target_link_libraries(
  yopo_mixed_scene_map_generator
  ${ROSCPP_LIBRARIES}
  ${PCL_LIBRARIES}
  ${YAML_CPP_LIBRARY}
)
