cmake_minimum_required(VERSION 3.10)
project(ImageRetrieval)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add src directory to include path
include_directories(${CMAKE_SOURCE_DIR}/src)

# Print OpenCV info for debugging
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Baseline matching executable
add_executable(baseline_match 
    src/baseline_match.cpp
    src/csv_util.cpp
)
target_link_libraries(baseline_match ${OpenCV_LIBS})

# Histogram matching executable
add_executable(histogram_match 
    src/histogram_match.cpp
    src/csv_util.cpp
)
target_link_libraries(histogram_match ${OpenCV_LIBS})

# Multi-histogram matching executable
add_executable(multi_histogram_match 
    src/multi_histogram_match.cpp
    src/csv_util.cpp
)
target_link_libraries(multi_histogram_match ${OpenCV_LIBS})

# Texture and color matching executable
add_executable(texture_color_match 
    src/texture_color_match.cpp
    src/csv_util.cpp
)
target_link_libraries(texture_color_match ${OpenCV_LIBS})

# Deep embedding matching executable
add_executable(deep_embedding_match 
    src/deep_embedding_match.cpp
    src/csv_util.cpp
)
target_link_libraries(deep_embedding_match ${OpenCV_LIBS})

# Custom sunset matching executable
add_executable(custom_sunset_match 
    src/custom_sunset_match.cpp
    src/csv_util.cpp
)
target_link_libraries(custom_sunset_match ${OpenCV_LIBS})

# Extension: Live DNN embedding matching
add_executable(live_dnn_match 
    src/live_dnn_match.cpp
)
target_link_libraries(live_dnn_match ${OpenCV_LIBS})
