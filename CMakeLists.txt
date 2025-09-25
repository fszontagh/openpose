cmake_minimum_required(VERSION 3.10)
project(OpenPoseCLI)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found")
endif()

include_directories(includes)

add_executable(openpose src/main.cpp src/openpose.cpp)
target_link_libraries(openpose ${OpenCV_LIBS})
