\# Image Retrieval Results



\## Task 1: Baseline Matching (7x7 Center Square + SSD)

\*\*Target: pic.1016.jpg\*\*

1\. pic.1016.jpg (0.00)

2\. pic.0986.jpg (14049.00)

3\. pic.0641.jpg (21756.00)

4\. pic.0547.jpg (49703.00)



\## Task 2: Histogram Matching (rg Chromaticity)

\*\*Target: pic.0164.jpg\*\*

1\. pic.0164.jpg (0.0000)

2\. pic.0080.jpg (0.3086)

3\. pic.1032.jpg (0.3724)

4\. pic.0461.jpg (0.4391)



\## Task 3: Multi-Histogram (Top/Bottom Halves RGB)

\*\*Target: pic.0274.jpg\*\*

1\. pic.0274.jpg (0.0000)

2\. pic.0273.jpg (0.3473)

3\. pic.1031.jpg (0.3751)

4\. pic.0409.jpg (0.3795)



\## Task 4: Texture + Color (Sobel + RGB)

\*\*Target: pic.0535.jpg\*\*

1\. pic.0535.jpg (0.0000)

2\. pic.0853.jpg (0.1678)

3\. pic.0605.jpg (0.1759)

4\. pic.0708.jpg (0.1762)



\## Task 5: Deep Network Embeddings (ResNet18 + Cosine Distance)



\### Target: pic.0893.jpg

\*\*Deep Embeddings:\*\*

1\. pic.0893.jpg (0.0000)

2\. pic.0897.jpg (0.1518)

3\. pic.0136.jpg (0.1762)



\*\*Histogram (rg):\*\*

1\. pic.0893.jpg (0.0000)

2\. pic.0899.jpg (0.1280)

3\. pic.0136.jpg (0.1421)



\*\*Multi-Histogram:\*\*

1\. pic.0893.jpg (0.0000)

2\. pic.0136.jpg (0.2659)

3\. pic.0897.jpg (0.3077)



\*\*Texture+Color:\*\*

1\. pic.0893.jpg (0.0000)

2\. pic.0899.jpg (0.1674)

3\. pic.0680.jpg (0.1675)



\### Target: pic.0164.jpg

\*\*Deep Embeddings:\*\*

1\. pic.0164.jpg (0.0000)

2\. pic.1032.jpg (0.2122)

3\. pic.0213.jpg (0.2128)



\*\*Histogram (rg):\*\*

1\. pic.0164.jpg (0.0000)

2\. pic.0080.jpg (0.3086)

3\. pic.1032.jpg (0.3724)



\*\*Multi-Histogram:\*\*

1\. pic.0164.jpg (0.0000)

2\. pic.0110.jpg (0.6148)

3\. pic.1032.jpg (0.6765)



\*\*Texture+Color:\*\*

1\. pic.0164.jpg (0.0000)

2\. pic.1032.jpg (0.3175)

3\. pic.0080.jpg (0.3455)



\## Observations

\- Different methods capture different aspects of similarity

\- Deep embeddings understand semantic content (what objects are in the image)

\- Histogram methods focus on color distribution

\- Multi-histogram considers spatial layout

\- Texture+color combines appearance and structure

\- pic.1032.jpg consistently appears as similar to pic.0164.jpg across methods

