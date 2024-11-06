cmake_minimum_required(VERSION 3.0.0)
project(ImageProcessing)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP REQUIRED)

add_executable(ImageProcessing main.cpp)

target_link_libraries(ImageProcessing PUBLIC ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(ImageProcessing PUBLIC OpenMP::OpenMP_CXX)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)