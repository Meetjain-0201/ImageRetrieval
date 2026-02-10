/*
  Baseline Image Matching using 7x7 center square and SSD
  
  Usage: baseline_match <target_image> <image_directory> <num_matches>
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include "csv_util.h"

using namespace cv;
using namespace std;

// Extract 7x7 center square from image as feature vector
vector<float> extractCenterSquare(Mat &image) {
    vector<float> features;
    
    // Get image center
    int centerY = image.rows / 2;
    int centerX = image.cols / 2;
    
    // Extract 7x7 region around center
    int halfSize = 3; // 7/2 = 3
    
    for(int i = centerY - halfSize; i <= centerY + halfSize; i++) {
        for(int j = centerX - halfSize; j <= centerX + halfSize; j++) {
            // Handle color images (3 channels: B, G, R)
            Vec3b pixel = image.at<Vec3b>(i, j);
            features.push_back(pixel[0]); // Blue
            features.push_back(pixel[1]); // Green
            features.push_back(pixel[2]); // Red
        }
    }
    
    return features;
}

// Compute Sum of Squared Differences between two feature vectors
float computeSSD(vector<float> &feat1, vector<float> &feat2) {
    float ssd = 0.0;
    
    if(feat1.size() != feat2.size()) {
        printf("Error: Feature vectors have different sizes!\n");
        return -1.0;
    }
    
    for(int i = 0; i < feat1.size(); i++) {
        float diff = feat1[i] - feat2[i];
        ssd += diff * diff;
    }
    
    return ssd;
}

// Structure to hold image filename and its distance from target
struct ImageMatch {
    string filename;
    float distance;
    
    bool operator<(const ImageMatch &other) const {
        return distance < other.distance;
    }
};

int main(int argc, char *argv[]) {
    
    // Check arguments
    if(argc < 4) {
        printf("Usage: %s <target_image> <image_directory> <num_matches>\n", argv[0]);
        printf("Example: %s images/pic.1016.jpg images 5\n", argv[0]);
        return -1;
    }
    
    char *targetImagePath = argv[1];
    char *imageDir = argv[2];
    int numMatches = atoi(argv[3]);
    
    // Load target image
    Mat targetImage = imread(targetImagePath);
    if(targetImage.empty()) {
        printf("Error: Could not load target image: %s\n", targetImagePath);
        return -1;
    }
    
    printf("Target image: %s (%d x %d)\n", targetImagePath, targetImage.cols, targetImage.rows);
    
    // Extract features from target image
    vector<float> targetFeatures = extractCenterSquare(targetImage);
    printf("Extracted %lu features from target image\n", targetFeatures.size());
    
    // Open directory
    DIR *dirp = opendir(imageDir);
    if(dirp == NULL) {
        printf("Error: Cannot open directory %s\n", imageDir);
        return -1;
    }
    
    // Process all images in directory
    vector<ImageMatch> matches;
    struct dirent *dp;
    char buffer[512];
    
    printf("\nProcessing images in directory: %s\n", imageDir);
    
    while((dp = readdir(dirp)) != NULL) {
        // Check if file is an image
        if(strstr(dp->d_name, ".jpg") || 
           strstr(dp->d_name, ".png") || 
           strstr(dp->d_name, ".ppm") || 
           strstr(dp->d_name, ".tif")) {
            
            // Build full path
            strcpy(buffer, imageDir);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            // Load image
            Mat image = imread(buffer);
            if(image.empty()) {
                printf("Warning: Could not load %s\n", buffer);
                continue;
            }
            
            // Extract features
            vector<float> features = extractCenterSquare(image);
            
            // Compute distance
            float distance = computeSSD(targetFeatures, features);
            
            // Store result
            ImageMatch match;
            match.filename = string(dp->d_name);
            match.distance = distance;
            matches.push_back(match);
            
            printf("  %s: distance = %.2f\n", dp->d_name, distance);
        }
    }
    
    closedir(dirp);
    
    // Sort matches by distance (ascending)
    sort(matches.begin(), matches.end());
    
    // Display top N matches
    printf("\n=== Top %d matches ===\n", numMatches);
    for(int i = 0; i < min(numMatches, (int)matches.size()); i++) {
        printf("%d. %s (distance: %.2f)\n", i+1, matches[i].filename.c_str(), matches[i].distance);
    }
    
    return 0;
}