cmake_minimum_required(VERSION 3.0.0)
project(ImageProcessing)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(ImageProcessing main.cpp)

target_link_libraries(ImageProcessing ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)