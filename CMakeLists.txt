cmake_minimum_required(VERSION 3.10)
project(pose_cli)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found")
endif()


add_executable(pose_cli pose_cli.cpp)
target_link_libraries(pose_cli ${OpenCV_LIBS})
