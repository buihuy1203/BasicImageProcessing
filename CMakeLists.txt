cmake_minimum_required(VERSION 3.25)
project(ImageProcessing)

include(CTest)
enable_testing()

add_executable(ImageProcessing main.cpp)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP REQUIRED)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCL_LIBRARIES})

target_link_libraries(ImageProcessing PUBLIC ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(ImageProcessing PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(ImageProcessing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)