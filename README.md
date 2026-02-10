\# Content-Based Image Retrieval



Computer Vision Project - Image Retrieval using Classic Features and Deep Network Embeddings



\## Overview

This project implements multiple methods for content-based image retrieval:

1\. \*\*Baseline Matching\*\*: 7x7 center square with SSD

2\. \*\*Histogram Matching\*\*: rg chromaticity histogram with histogram intersection

3\. \*\*Multi-Histogram Matching\*\*: Top/bottom RGB histograms

4\. \*\*Texture + Color Matching\*\*: Sobel gradients + RGB histogram

5\. \*\*Deep Network Embeddings\*\*: ResNet18 features with cosine distance

6\. \*\*Custom Sunset/Warm Scene Detector\*\*: Domain-specific features combining warm color detection, vertical gradients, edge density, and DNN embeddings



\## Requirements

\- C++11 or higher

\- CMake 3.10+

\- OpenCV 4.x

\- MinGW-w64 (Windows) or GCC (Linux/Mac)



\## Build Instructions

```bash

mkdir build

cd build

cmake ..

cmake --build .

```



\## Usage



\### Baseline Matching

```bash

baseline\_match <target\_image> <image\_directory> <num\_matches>

```



\### Histogram Matching

```bash

histogram\_match <target\_image> <image\_directory> <num\_matches>

```



\### Multi-Histogram Matching

```bash

multi\_histogram\_match <target\_image> <image\_directory> <num\_matches>

```



\### Texture + Color Matching

```bash

texture\_color\_match <target\_image> <image\_directory> <num\_matches>

```



\### Deep Network Embeddings

```bash

deep\_embedding\_match <target\_image\_name> <csv\_file> <num\_matches>

```



\### Custom Sunset/Warm Scene Detector

```bash

custom\_sunset\_match <target\_image> <image\_directory> <csv\_file> <num\_matches>

```



\## Project Structure

```

ImageRetrieval/

├── src/

│   ├── baseline\_match.cpp

│   ├── histogram\_match.cpp

│   ├── multi\_histogram\_match.cpp

│   ├── texture\_color\_match.cpp

│   ├── deep\_embedding\_match.cpp

│   ├── custom\_sunset\_match.cpp

│   ├── csv\_util.cpp

│   └── csv\_util.h

├── build/

├── images/

│   └── olympus/

├── data/

│   └── ResNet18\_olym.csv

├── CMakeLists.txt

├── README.md

├── RESULTS.md

├── COMPARISON\_ANALYSIS.md

└── CUSTOM\_FEATURE\_REPORT.md

```



\## Documentation

\- \*\*\[RESULTS.md](RESULTS.md)\*\* - Detailed matching results for each method

\- \*\*\[COMPARISON\_ANALYSIS.md](COMPARISON\_ANALYSIS.md)\*\* - Comparative analysis of DNN vs classic features

\- \*\*\[CUSTOM\_FEATURE\_REPORT.md](CUSTOM\_FEATURE\_REPORT.md)\*\* - Custom sunset detector design and evaluation



\## Key Findings



\### General Image Retrieval

\- \*\*DNN embeddings excel at semantic similarity\*\* (finding same objects/scenes)

\- \*\*Classic features better for appearance-based matching\*\* (color, texture)

\- \*\*Hybrid approaches recommended\*\* for robust retrieval systems

\- \*\*No single method is always best\*\* - depends on query intent



\### Custom Sunset/Warm Scene Detection

\- \*\*Custom features outperform generic methods\*\* for domain-specific tasks

\- \*\*Vertical gradient detection\*\* (warm top/cool bottom) is key discriminator for sunsets

\- \*\*Combining 60% hand-crafted + 40% DNN features\*\* provides optimal performance

\- \*\*Interpretability matters\*\* - can explain why images match based on measurable features



\## Example Results



\### Sunset Detection (pic.0733.jpg)

Custom detector successfully identifies sunset-like scenes based on:

\- \*\*Warm color score\*\*: 0.046 (presence of red/orange/yellow)

\- \*\*Vertical gradient\*\*: 269.36 (warm top, cool bottom)

\- \*\*Edge density\*\*: 0.208 (smooth sky gradients)

\- \*\*DNN context\*\*: Outdoor scene validation



Top matches all share strong positive gradients (235-296), while least similar images show negative gradients (-188 to -244), demonstrating the discriminative power of custom features.



\## Technologies Used

\- \*\*C++\*\* for core image processing

\- \*\*OpenCV 4.13\*\* for computer vision operations

\- \*\*CMake\*\* for cross-platform building

\- \*\*ResNet18\*\* pre-trained embeddings for semantic features



\## Extensions



\### Extension 1: Live DNN Embedding Computation

Implemented real-time ResNet18 embedding computation using OpenCV's DNN module instead of pre-computed CSV files. See \[EXTENSIONS.md](EXTENSIONS.md) for detailed analysis.



\*\*Key Achievement\*\*: Successfully reproduced DNN embeddings with consistent top-3 matches (pic.0136.jpg, pic.0897.jpg) compared to pre-computed method, validating correct implementation.



\*\*Usage:\*\*

```bash

live\_dnn\_match <target\_image> <image\_directory> <onnx\_model> <num\_matches>

```



\## Author

Meetjain-0201



\## Project Completion

All required tasks implemented and tested:

\- ✅ Baseline matching with exact required results

\- ✅ Histogram matching with multiple color spaces

\- ✅ Multi-histogram spatial features

\- ✅ Texture and color combined features

\- ✅ Deep network embeddings with comparisons

\- ✅ Custom domain-specific feature design and evaluation

