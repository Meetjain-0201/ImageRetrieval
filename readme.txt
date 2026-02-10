CONTENT-BASED IMAGE RETRIEVAL - PROJECT 2
==========================================

Student: Meet Jain
GitHub: https://github.com/Meetjain-0201/ImageRetrieval

SYSTEM INFORMATION
------------------
Operating System: Windows 11 (Version 10.0.26200.7623)
IDE: None (used Notepad for editing, command line for compilation)
Compiler: MinGW-w64 GCC 15.2.0
CMake Version: 4.2.3
OpenCV Version: 4.13.0

BUILD INSTRUCTIONS
------------------
1. Navigate to project directory
2. Create build directory:
   mkdir build
   cd build

3. Configure with CMake:
   cmake ..

4. Build all executables:
   cmake --build .

RUNNING THE PROGRAMS
--------------------

1. Baseline Matching:
   baseline_match.exe <target_image> <image_directory> <num_matches>
   Example: baseline_match.exe ..\images\olympus\pic.1016.jpg ..\images\olympus 5

2. Histogram Matching:
   histogram_match.exe <target_image> <image_directory> <num_matches>
   Example: histogram_match.exe ..\images\olympus\pic.0164.jpg ..\images\olympus 5

3. Multi-Histogram Matching:
   multi_histogram_match.exe <target_image> <image_directory> <num_matches>
   Example: multi_histogram_match.exe ..\images\olympus\pic.0274.jpg ..\images\olympus 5

4. Texture + Color Matching:
   texture_color_match.exe <target_image> <image_directory> <num_matches>
   Example: texture_color_match.exe ..\images\olympus\pic.0535.jpg ..\images\olympus 5

5. Deep Network Embeddings:
   deep_embedding_match.exe <target_image_name> <csv_file> <num_matches>
   Example: deep_embedding_match.exe pic.0893.jpg ..\data\ResNet18_olym.csv 5
   Note: Use just the filename (not full path) for target image

6. Custom Sunset/Warm Scene Detector:
   custom_sunset_match.exe <target_image> <image_directory> <csv_file> <num_matches>
   Example: custom_sunset_match.exe ..\images\olympus\pic.0733.jpg ..\images\olympus ..\data\ResNet18_olym.csv 10

EXTENSION: Live DNN Embedding Computation
------------------------------------------
7. Live DNN Matching (Extension):
   live_dnn_match.exe <target_image> <image_directory> <onnx_model> <num_matches>
   Example: live_dnn_match.exe ..\images\olympus\pic.0893.jpg ..\images\olympus ..\models\resnet18-v2-7.onnx 5
   
   Note: This computes ResNet18 embeddings in real-time instead of using pre-computed CSV.
   Processing 1106 images takes approximately 60-90 seconds.
   
   Validation: Results match pre-computed CSV method (pic.0136.jpg and pic.0897.jpg 
   both appear in top-3 for test image pic.0893.jpg).

PROJECT STRUCTURE
-----------------
ImageRetrieval/
├── src/
│   ├── baseline_match.cpp          - Task 1
│   ├── histogram_match.cpp         - Task 2
│   ├── multi_histogram_match.cpp   - Task 3
│   ├── texture_color_match.cpp     - Task 4
│   ├── deep_embedding_match.cpp    - Task 5
│   ├── custom_sunset_match.cpp     - Task 7
│   ├── live_dnn_match.cpp          - Extension
│   ├── csv_util.cpp                - Utility functions
│   └── csv_util.h                  - Header file
├── build/                          - Compiled executables
├── images/olympus/                 - Image database (1106 images)
├── data/ResNet18_olym.csv         - Pre-computed DNN features
├── models/resnet18-v2-7.onnx      - ResNet18 ONNX model
└── CMakeLists.txt                 - Build configuration

REQUIRED FILES INCLUDED
-----------------------
All .cpp source files (7 programs)
csv_util.h header file
CMakeLists.txt
PDF report with all required results
readme.txt (this file)

TESTING NOTES
-------------
- All required test cases produce exact expected results
- Task 1 (pic.1016.jpg): Matches pic.0986, pic.0641, pic.0547 exactly
- Task 2 (pic.0164.jpg): Matches pic.0080, pic.1032, pic.0461 as expected
- Task 3 (pic.0274.jpg): Matches pic.0273, pic.1031, pic.0409 as expected
- All distance calculations verified (self-distance = 0.00 in all methods)

EXTENSION HIGHLIGHTS
--------------------
- Live DNN embedding computation using OpenCV DNN module
- Real-time ResNet18 inference with proper ImageNet preprocessing
- Validates pre-computed embeddings (top-3 matches consistent)
- Demonstrates understanding of neural network forward pass
- More flexible but slower than CSV-based approach

TIME TRAVEL DAYS
----------------
Not using any time travel days for this submission.

ADDITIONAL NOTES
----------------
- GitHub repository contains full project history: https://github.com/Meetjain-0201/ImageRetrieval
- All documentation (RESULTS.md, COMPARISON_ANALYSIS.md, CUSTOM_FEATURE_REPORT.md, 
  EXTENSIONS.md) included in repository
- Code is well-commented and follows consistent style
- No external code libraries used beyond OpenCV and provided utilities