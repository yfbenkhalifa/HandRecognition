cmake_minimum_required(VERSION 3.0)
project(main)
set(SOURCE_FILES src/main.cpp)

include_directories(lib/preprocess)
include_directories(lib/preprocess)
add_executable(${PROJECT_NAME} src/main.cpp)
# Add lib subdirectory (to build our libraries)
add_subdirectory(lib/preprocess)


