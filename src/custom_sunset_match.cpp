/*
  Custom Sunset/Warm Scene Matching
  
  Combines:
  - Warm color detection in upper image region
  - Color gradient analysis
  - Texture smoothness (low edge density)
  - Deep network embeddings
  
  Usage: custom_sunset_match <target_image> <image_directory> <csv_file> <num_matches>
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <dirent.h>
#include "csv_util.h"

using namespace cv;
using namespace std;

// Extract warm color percentage from upper portion of image
float computeWarmColorScore(Mat &image) {
    int warmPixels = 0;
    int totalPixels = 0;
    
    // Focus on upper 60% of image (where sky/sunset typically is)
    int endRow = (int)(image.rows * 0.6);
    
    for(int i = 0; i < endRow; i++) {
        for(int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            
            float b = pixel[0];
            float g = pixel[1];
            float r = pixel[2];
            
            // Check if pixel is "warm" (red/orange/yellow dominant)
            // Warm colors: R > G > B, with R being significantly higher
            if(r > g && g >= b && r > 100) {
                // Additional check: R should be at least 20% more than G
                if(r > g * 1.2) {
                    warmPixels++;
                }
            }
            
            totalPixels++;
        }
    }
    
    return (float)warmPixels / totalPixels;
}

// Compute vertical color gradient (sunset transitions from warm to cool)
float computeVerticalGradient(Mat &image) {
    // Split image into top third and bottom third
    int topStart = 0;
    int topEnd = image.rows / 3;
    int bottomStart = (2 * image.rows) / 3;
    int bottomEnd = image.rows;
    
    // Average colors in top third
    float topR = 0, topG = 0, topB = 0;
    int topCount = 0;
    
    for(int i = topStart; i < topEnd; i++) {
        for(int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            topB += pixel[0];
            topG += pixel[1];
            topR += pixel[2];
            topCount++;
        }
    }
    
    topR /= topCount;
    topG /= topCount;
    topB /= topCount;
    
    // Average colors in bottom third
    float bottomR = 0, bottomG = 0, bottomB = 0;
    int bottomCount = 0;
    
    for(int i = bottomStart; i < bottomEnd; i++) {
        for(int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            bottomB += pixel[0];
            bottomG += pixel[1];
            bottomR += pixel[2];
            bottomCount++;
        }
    }
    
    bottomR /= bottomCount;
    bottomG /= bottomCount;
    bottomB /= bottomCount;
    
    // Compute gradient: sunsets have warmer top, cooler bottom
    float warmGradient = (topR - bottomR) + (topG - bottomG) * 0.5;
    
    return warmGradient;
}

// Compute edge density (sunsets are smooth, not busy)
float computeEdgeDensity(Mat &image) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    Mat edges;
    Canny(gray, edges, 50, 150);
    
    int edgePixels = countNonZero(edges);
    int totalPixels = edges.rows * edges.cols;
    
    return (float)edgePixels / totalPixels;
}

// Normalize vector
vector<float> normalizeVector(vector<float> &vec) {
    vector<float> normalized;
    float norm = 0.0;
    
    for(int i = 0; i < vec.size(); i++) {
        norm += vec[i] * vec[i];
    }
    norm = sqrt(norm);
    
    if(norm > 0) {
        for(int i = 0; i < vec.size(); i++) {
            normalized.push_back(vec[i] / norm);
        }
    } else {
        normalized = vec;
    }
    
    return normalized;
}

// Compute cosine distance for DNN embeddings
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

// Combined custom distance metric for sunset detection
float computeSunsetDistance(float warmScore1, float gradient1, float edgeDensity1, vector<float> &dnn1,
                             float warmScore2, float gradient2, float edgeDensity2, vector<float> &dnn2) {
    
    // Warm color difference (most important for sunsets)
    float warmDiff = fabs(warmScore1 - warmScore2);
    
    // Gradient difference
    float gradDiff = fabs(gradient1 - gradient2);
    
    // Edge density difference (prefer smooth images)
    float edgeDiff = fabs(edgeDensity1 - edgeDensity2);
    
    // DNN embedding distance
    float dnnDist = cosineDistance(dnn1, dnn2);
    
    // Weighted combination
    // Weights: warm=40%, gradient=20%, smoothness=10%, DNN=30%
    float combinedDistance = 
        0.40 * warmDiff +
        0.20 * (gradDiff / 50.0) +  // Normalize gradient to 0-1 range
        0.10 * edgeDiff +
        0.30 * dnnDist;
    
    return combinedDistance;
}

struct ImageFeatures {
    string filename;
    float warmScore;
    float gradient;
    float edgeDensity;
    vector<float> dnnEmbedding;
    float distance;
    
    bool operator<(const ImageFeatures &other) const {
        return distance < other.distance;
    }
};

int main(int argc, char *argv[]) {
    
    if(argc < 5) {
        printf("Usage: %s <target_image> <image_directory> <csv_file> <num_matches>\n", argv[0]);
        printf("Example: %s images/pic.0365.jpg images data/ResNet18_olym.csv 10\n", argv[0]);
        return -1;
    }
    
    char *targetImagePath = argv[1];
    char *imageDir = argv[2];
    char *csvFile = argv[3];
    int numMatches = atoi(argv[4]);
    
    // Load DNN embeddings
    vector<char *> embeddingFilenames;
    vector<vector<float>> embeddings;
    
    printf("Loading DNN embeddings from: %s\n", csvFile);
    if(read_image_data_csv(csvFile, embeddingFilenames, embeddings, 0) != 0) {
        printf("Error: Failed to read CSV file\n");
        return -1;
    }
    
    // Load target image
    Mat targetImage = imread(targetImagePath);
    if(targetImage.empty()) {
        printf("Error: Could not load target image: %s\n", targetImagePath);
        return -1;
    }
    
    printf("\n=== Analyzing Target Image ===\n");
    printf("Target: %s (%d x %d)\n", targetImagePath, targetImage.cols, targetImage.rows);
    
    // Extract target image filename
    string targetPath(targetImagePath);
    string targetFilename = targetPath.substr(targetPath.find_last_of("/\\") + 1);
    
    // Compute target features
    float targetWarm = computeWarmColorScore(targetImage);
    float targetGrad = computeVerticalGradient(targetImage);
    float targetEdge = computeEdgeDensity(targetImage);
    
    printf("Warm color score: %.4f\n", targetWarm);
    printf("Vertical gradient: %.2f\n", targetGrad);
    printf("Edge density: %.4f\n", targetEdge);
    
    // Find target DNN embedding
    vector<float> targetDNN;
    for(int i = 0; i < embeddingFilenames.size(); i++) {
        if(strcmp(embeddingFilenames[i], targetFilename.c_str()) == 0) {
            targetDNN = embeddings[i];
            break;
        }
    }
    
    if(targetDNN.empty()) {
        printf("Warning: DNN embedding not found for target, using zeros\n");
        targetDNN.resize(512, 0.0);
    }
    
    // Process all images
    DIR *dirp = opendir(imageDir);
    if(dirp == NULL) {
        printf("Error: Cannot open directory %s\n", imageDir);
        return -1;
    }
    
    vector<ImageFeatures> results;
    struct dirent *dp;
    char buffer[512];
    
    printf("\n=== Processing Database Images ===\n");
    
    while((dp = readdir(dirp)) != NULL) {
        if(strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png")) {
            
            strcpy(buffer, imageDir);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            Mat image = imread(buffer);
            if(image.empty()) continue;
            
            // Compute features
            ImageFeatures feat;
            feat.filename = string(dp->d_name);
            feat.warmScore = computeWarmColorScore(image);
            feat.gradient = computeVerticalGradient(image);
            feat.edgeDensity = computeEdgeDensity(image);
            
            // Get DNN embedding
            for(int i = 0; i < embeddingFilenames.size(); i++) {
                if(strcmp(embeddingFilenames[i], dp->d_name) == 0) {
                    feat.dnnEmbedding = embeddings[i];
                    break;
                }
            }
            
            if(feat.dnnEmbedding.empty()) {
                feat.dnnEmbedding.resize(512, 0.0);
            }
            
            // Compute distance
            feat.distance = computeSunsetDistance(
                targetWarm, targetGrad, targetEdge, targetDNN,
                feat.warmScore, feat.gradient, feat.edgeDensity, feat.dnnEmbedding
            );
            
            results.push_back(feat);
        }
    }
    
    closedir(dirp);
    
    // Sort by distance
    sort(results.begin(), results.end());
    
    // Display top matches
    printf("\n=== Top %d Sunset Matches ===\n", numMatches);
    for(int i = 0; i < min(numMatches, (int)results.size()); i++) {
        printf("%d. %s (dist: %.4f, warm: %.3f, grad: %.1f, edge: %.3f)\n", 
               i+1, results[i].filename.c_str(), results[i].distance,
               results[i].warmScore, results[i].gradient, results[i].edgeDensity);
    }
    
    // Display least similar (bottom 5)
    printf("\n=== Least Similar Images (Bottom 5) ===\n");
    int start = max(0, (int)results.size() - 5);
    for(int i = start; i < results.size(); i++) {
        printf("%d. %s (dist: %.4f, warm: %.3f, grad: %.1f, edge: %.3f)\n", 
               i+1, results[i].filename.c_str(), results[i].distance,
               results[i].warmScore, results[i].gradient, results[i].edgeDensity);
    }
    
    // Clean up
    for(int i = 0; i < embeddingFilenames.size(); i++) {
        delete[] embeddingFilenames[i];
    }
    
    return 0;
}