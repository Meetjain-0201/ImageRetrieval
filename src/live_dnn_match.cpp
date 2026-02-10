/*
  Live DNN Embedding Matching - Computes ResNet18 embeddings on-the-fly
  
  Extension: Instead of using pre-computed CSV, this program loads the ResNet18
  ONNX model and computes embeddings for each image during matching.
  
  Usage: live_dnn_match <target_image> <image_directory> <onnx_model> <num_matches>
*/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <dirent.h>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Compute ResNet18 embedding for an image
int getEmbedding(Mat &src, Mat &embedding, Net &net) {
    const int ORNet_size = 224;
    Mat blob;
    
    // ImageNet normalization and resize to 224x224
    dnn::blobFromImage(src,
                       blob,
                       (1.0/255.0) * (1/0.226),
                       Size(ORNet_size, ORNet_size),
                       Scalar(124, 116, 104),
                       true,
                       false,
                       CV_32F);
    
    net.setInput(blob);
    embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");
    
    return 0;
}

// Convert Mat to vector<float>
vector<float> matToVector(Mat &mat) {
    vector<float> vec;
    for(int i = 0; i < mat.total(); i++) {
        vec.push_back(mat.at<float>(0, i));
    }
    return vec;
}

// Normalize vector by L2 norm
vector<float> normalizeVector(vector<float> &vec) {
    vector<float> normalized;
    float norm = 0.0;
    
    for(float val : vec) {
        norm += val * val;
    }
    norm = sqrt(norm);
    
    if(norm > 0) {
        for(float val : vec) {
            normalized.push_back(val / norm);
        }
    } else {
        normalized = vec;
    }
    
    return normalized;
}

// Compute cosine distance
float cosineDistance(vector<float> &vec1, vector<float> &vec2) {
    vector<float> norm1 = normalizeVector(vec1);
    vector<float> norm2 = normalizeVector(vec2);
    
    float dotProd = 0.0;
    for(int i = 0; i < norm1.size(); i++) {
        dotProd += norm1[i] * norm2[i];
    }
    
    if(dotProd > 1.0) dotProd = 1.0;
    if(dotProd < -1.0) dotProd = -1.0;
    
    return 1.0 - dotProd;
}

struct ImageMatch {
    string filename;
    float distance;
    
    bool operator<(const ImageMatch &other) const {
        return distance < other.distance;
    }
};

int main(int argc, char *argv[]) {
    
    if(argc < 5) {
        printf("Usage: %s <target_image> <image_directory> <onnx_model> <num_matches>\n", argv[0]);
        printf("Example: %s images/pic.0893.jpg images models/resnet18-v2-7.onnx 10\n", argv[0]);
        return -1;
    }
    
    char *targetImagePath = argv[1];
    char *imageDir = argv[2];
    char *modelPath = argv[3];
    int numMatches = atoi(argv[4]);
    
    // Load ResNet18 network
    printf("Loading ResNet18 model from: %s\n", modelPath);
    Net net = readNet(modelPath);
    if(net.empty()) {
        printf("Error: Could not load network\n");
        return -1;
    }
    printf("Network loaded successfully!\n");
    
    // Print layer information
    vector<String> layerNames = net.getLayerNames();
    printf("Network has %lu layers\n", layerNames.size());
    
    // Load target image
    Mat targetImage = imread(targetImagePath);
    if(targetImage.empty()) {
        printf("Error: Could not load target image: %s\n", targetImagePath);
        return -1;
    }
    
    printf("\n=== Processing Target Image ===\n");
    printf("Target: %s (%d x %d)\n", targetImagePath, targetImage.cols, targetImage.rows);
    
    // Compute target embedding
    Mat targetEmbeddingMat;
    printf("Computing embedding for target image...\n");
    getEmbedding(targetImage, targetEmbeddingMat, net);
    vector<float> targetEmbedding = matToVector(targetEmbeddingMat);
    printf("Target embedding computed: %lu dimensions\n", targetEmbedding.size());
    
    // Process all images in directory
    DIR *dirp = opendir(imageDir);
    if(dirp == NULL) {
        printf("Error: Cannot open directory %s\n", imageDir);
        return -1;
    }
    
    vector<ImageMatch> matches;
    struct dirent *dp;
    char buffer[512];
    int processedCount = 0;
    
    printf("\n=== Processing Database Images ===\n");
    
    while((dp = readdir(dirp)) != NULL) {
        if(strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png")) {
            
            strcpy(buffer, imageDir);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            Mat image = imread(buffer);
            if(image.empty()) {
                printf("Warning: Could not load %s\n", buffer);
                continue;
            }
            
            // Compute embedding for this image
            Mat embeddingMat;
            getEmbedding(image, embeddingMat, net);
            vector<float> embedding = matToVector(embeddingMat);
            
            // Compute distance
            float distance = cosineDistance(targetEmbedding, embedding);
            
            ImageMatch match;
            match.filename = string(dp->d_name);
            match.distance = distance;
            matches.push_back(match);
            
            processedCount++;
            if(processedCount % 100 == 0) {
                printf("Processed %d images...\n", processedCount);
            }
        }
    }
    
    closedir(dirp);
    
    printf("Total images processed: %d\n", processedCount);
    
    // Sort by distance
    sort(matches.begin(), matches.end());
    
    // Display results
    printf("\n=== Top %d Matches (Live DNN Embeddings) ===\n", numMatches);
    for(int i = 0; i < min(numMatches, (int)matches.size()); i++) {
        printf("%d. %s (distance: %.4f)\n", i+1, matches[i].filename.c_str(), matches[i].distance);
    }
    
    printf("\n=== Performance Note ===\n");
    printf("This extension computes embeddings in real-time using the ResNet18 ONNX model.\n");
    printf("Pro: No need for pre-computed CSV files\n");
    printf("Con: Slower than using cached embeddings (but more flexible!)\n");
    
    return 0;
}