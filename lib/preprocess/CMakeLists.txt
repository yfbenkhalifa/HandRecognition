cmake_minimum_required(VERSION 3.0)
project(preprocess)

# Add INTERFACE library, thus, header-only
add_library(${PROJECT_NAME} src/preprocess.cpp)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
