set (CMAKE_CXX_STANDARD 14)

cmake_minimum_required(VERSION 3.0.0)
project(hand_recognition)

find_package(OpenCV REQUIRED)
# Multithreading library
find_package(TBB REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${PYTHON_INCLUDE_DIRS})

include_directories(include)
file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} TBB::tbb)