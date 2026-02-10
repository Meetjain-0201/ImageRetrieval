cmake_minimum_required(VERSION 3.10)
project(ImageRetrieval)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add src directory to include path
include_directories(${CMAKE_SOURCE_DIR}/src)

# Baseline matching executable
add_executable(baseline_match 
    src/baseline_match.cpp
    src/csv_util.cpp
)
target_link_libraries(baseline_match ${OpenCV_LIBS})