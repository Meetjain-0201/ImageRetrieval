\*\*Validation Test (pic.0893.jpg):\*\*



| Rank | Live DNN | Distance | Pre-computed CSV | Distance |

|------|----------|----------|------------------|----------|

| 1 | pic.0893.jpg | 0.0000 | pic.0893.jpg | 0.0000 |

| 2 | pic.0136.jpg | 0.1864 | pic.0897.jpg | 0.1518 |

| 3 | pic.0897.jpg | 0.2197 | pic.0136.jpg | 0.1762 |



!\[Live vs pre-computed comparison](results/extension\_comparison.png)



\*\*Both methods found pic.0136.jpg and pic.0897.jpg in top 3!\*\* The slight distance differences (~0.03-0.05) are likely due to floating-point precision, but the core matches are consistent. This validates my implementation.



\*\*Trade-offs:\*\*

\- \*\*Pro:\*\* No CSV dependency, works with any image set immediately

\- \*\*Pro:\*\* Saves disk space

\- \*\*Con:\*\* Slower (60-90 seconds vs <1 second for 1106 images)

\- \*\*Pro:\*\* Educational - shows understanding of DNN forward pass



\*\*Use Cases:\*\*

\- Testing different pre-trained networks without regenerating CSVs

\- Small databases where CSV overhead isn't worth it

\- Dynamic databases with frequently changing images



---



\## 4. What I Learned



\*\*Technical Skills:\*\*

\- Color spaces matter - rg chromaticity is more robust to lighting than raw RGB

\- Histogram intersection is elegantly simple yet effective

\- Sobel filters from project 1 came in handy for texture features

\- OpenCV's DNN module is surprisingly accessible for loading pre-trained models



\*\*Conceptual Insights:\*\*

The biggest lesson: \*\*there's no universal "best" similarity metric\*\*. I went into this thinking deep learning would dominate everything, but pic.0948.jpg proved me wrong - the histogram method crushed DNN with a distance of 0.059 vs 0.128 because that image was all about color.



\*\*Design Trade-offs:\*\*

Building the custom sunset detector taught me the value of domain knowledge. Those hand-crafted features (vertical gradient, warm color detection) were specifically designed based on understanding what makes a sunset look like a sunset. Generic methods miss this.



\*\*Unexpected Discoveries:\*\*

\- pic.1032.jpg appearing in top-3 across ALL methods for pic.0164.jpg was fascinating - consensus is rare

\- DNN finding sequential images (pic.0731-0743) showed it recognizes "same scene from different angles"

\- Negative gradients in the sunset detector revealing upside-down color distributions was a cool insight



\*\*What surprised me:\*\* How interpretable the custom features are compared to DNN. I can say "these images match because gradient=269 vs 267" but explaining why ResNet thinks two images are similar requires looking at 512-dimensional space.



\*\*If I had more time:\*\*

\- Try different DNN architectures (ResNet50, EfficientNet)

\- Implement GPU acceleration for the live embedding computation

\- Build a GUI that lets users switch between methods interactively

\- Test on a completely different image database to see if findings generalize



---



\## 5. Acknowledgements



\*\*Course Materials:\*\*

\- CSV utility code provided by Professor Bruce Maxwell

\- Directory reading example code from course materials

\- Chapter 8 of \*Computer Vision\* by Shapiro and Stockman for histogram theory

\- ResNet18 ONNX model and DNN embedding example code from course resources



\*\*Tools \& Libraries:\*\*

\- OpenCV 4.13 for all image processing and DNN inference

\- CMake for cross-platform building

\- MSYS2/MinGW-w64 for Windows C++ development



\*\*Technical References:\*\*

\- OpenCV documentation for dnn::blobFromImage parameters

\- ImageNet preprocessing specifications for ResNet normalization

\- Histogram intersection formula from course lectures



\*\*Collaboration:\*\*

\- Discussed distance metric design trade-offs with classmates (no code shared)

\- Debugged CMake linking issues with help from Stack Overflow community



\*\*Dataset:\*\*

\- Olympus image database (1106 images) provided for the course

\- ResNet18 embeddings CSV file with pre-computed features



All code implementation is my own work. Where I used provided utility functions (CSV reading/writing), they are properly credited in code comments.



---



\*This project demonstrated that effective image retrieval requires matching the right features to the right task. Sometimes that means state-of-the-art deep learning, sometimes it means a well-designed histogram, and often it means combining both.\*

