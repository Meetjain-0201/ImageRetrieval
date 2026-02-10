/*
  Texture and Color Matching using RGB histogram + Sobel gradient magnitude histogram
  
  Usage: texture_color_match <target_image> <image_directory> <num_matches>
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <cmath>
#include "csv_util.h"

using namespace cv;
using namespace std;

// Compute 3D RGB histogram for entire image
vector<float> computeRGBHistogram(Mat &image, int bins = 8) {
    vector<float> histogram(bins * bins * bins, 0.0);
    
    int totalPixels = 0;
    
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            
            // Map to bin indices [0, bins-1]
            int r_bin = (r * bins) / 256;
            int g_bin = (g * bins) / 256;
            int b_bin = (b * bins) / 256;
            
            // Clamp to valid range
            if(r_bin >= bins) r_bin = bins - 1;
            if(g_bin >= bins) g_bin = bins - 1;
            if(b_bin >= bins) b_bin = bins - 1;
            
            // Compute linear index
            int binIndex = r_bin * bins * bins + g_bin * bins + b_bin;
            histogram[binIndex]++;
            totalPixels++;
        }
    }
    
    // Normalize histogram
    for(int i = 0; i < histogram.size(); i++) {
        if(totalPixels > 0) {
            histogram[i] = histogram[i] / totalPixels;
        }
    }
    
    return histogram;
}

// Compute Sobel gradient magnitude and create histogram
vector<float> computeTextureHistogram(Mat &image, int bins = 16) {
    // Convert to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    // Apply Sobel filters
    Mat sobelX, sobelY;
    Sobel(gray, sobelX, CV_16S, 1, 0, 3);
    Sobel(gray, sobelY, CV_16S, 0, 1, 3);
    
    // Compute gradient magnitude
    Mat magnitude(gray.rows, gray.cols, CV_32F);
    
    for(int i = 0; i < gray.rows; i++) {
        for(int j = 0; j < gray.cols; j++) {
            float gx = sobelX.at<short>(i, j);
            float gy = sobelY.at<short>(i, j);
            magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);
        }
    }
    
    // Find max magnitude for normalization
    double minVal, maxVal;
    minMaxLoc(magnitude, &minVal, &maxVal);
    
    // Create histogram of gradient magnitudes
    vector<float> histogram(bins, 0.0);
    int totalPixels = 0;
    
    for(int i = 0; i < magnitude.rows; i++) {
        for(int j = 0; j < magnitude.cols; j++) {
            float mag = magnitude.at<float>(i, j);
            
            // Normalize magnitude to [0, bins-1]
            int bin = (int)((mag / maxVal) * bins);
            
            // Clamp to valid range
            if(bin >= bins) bin = bins - 1;
            
            histogram[bin]++;
            totalPixels++;
        }
    }
    
    // Normalize histogram
    for(int i = 0; i < histogram.size(); i++) {
        if(totalPixels > 0) {
            histogram[i] = histogram[i] / totalPixels;
        }
    }
    
    return histogram;
}

// Compute histogram intersection
float histogramIntersection(vector<float> &hist1, vector<float> &hist2) {
    if(hist1.size() != hist2.size()) {
        printf("Error: Histograms have different sizes!\n");
        return 0.0;
    }
    
    float intersection = 0.0;
    
    for(int i = 0; i < hist1.size(); i++) {
        intersection += min(hist1[i], hist2[i]);
    }
    
    return intersection;
}

// Compute combined distance with equal weighting
float computeCombinedDistance(vector<float> &colorHist1, vector<float> &textureHist1,
                               vector<float> &colorHist2, vector<float> &textureHist2) {
    // Compute color histogram intersection
    float colorIntersection = histogramIntersection(colorHist1, colorHist2);
    
    // Compute texture histogram intersection
    float textureIntersection = histogramIntersection(textureHist1, textureHist2);
    
    // Equal weighting: average the two intersections
    float avgIntersection = (colorIntersection + textureIntersection) / 2.0;
    
    // Return distance (1 - intersection)
    return 1.0 - avgIntersection;
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
        printf("Example: %s images/pic.0535.jpg images 5\n", argv[0]);
        return -1;
    }
    
    char *targetImagePath = argv[1];
    char *imageDir = argv[2];
    int numMatches = atoi(argv[3]);
    int colorBins = 8;    // 8x8x8 RGB histogram
    int textureBins = 16; // 16 bins for gradient magnitude
    
    // Load target image
    Mat targetImage = imread(targetImagePath);
    if(targetImage.empty()) {
        printf("Error: Could not load target image: %s\n", targetImagePath);
        return -1;
    }
    
    printf("Target image: %s (%d x %d)\n", targetImagePath, targetImage.cols, targetImage.rows);
    printf("Using %dx%dx%d RGB histogram and %d-bin texture histogram\n", 
           colorBins, colorBins, colorBins, textureBins);
    
    // Extract features from target image
    vector<float> targetColorHist = computeRGBHistogram(targetImage, colorBins);
    vector<float> targetTextureHist = computeTextureHistogram(targetImage, textureBins);
    printf("Computed color histogram: %lu bins\n", targetColorHist.size());
    printf("Computed texture histogram: %lu bins\n", targetTextureHist.size());
    
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
            
            // Compute features
            vector<float> colorHist = computeRGBHistogram(image, colorBins);
            vector<float> textureHist = computeTextureHistogram(image, textureBins);
            
            // Compute distance
            float distance = computeCombinedDistance(targetColorHist, targetTextureHist,
                                                      colorHist, textureHist);
            
            // Store result
            ImageMatch match;
            match.filename = string(dp->d_name);
            match.distance = distance;
            matches.push_back(match);
        }
    }
    
    closedir(dirp);
    
    // Sort matches by distance (ascending)
    sort(matches.begin(), matches.end());
    
    // Display top N matches
    printf("\n=== Top %d matches (Texture + Color) ===\n", numMatches);
    for(int i = 0; i < min(numMatches, (int)matches.size()); i++) {
        printf("%d. %s (distance: %.4f)\n", i+1, matches[i].filename.c_str(), matches[i].distance);
    }
    
    return 0;
}