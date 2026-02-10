/*
  Histogram Matching using rg chromaticity histogram and histogram intersection
  
  Usage: histogram_match <target_image> <image_directory> <num_matches>
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

// Compute 2D rg chromaticity histogram
// bins: number of bins for each dimension (default 16)
vector<float> computeRGHistogram(Mat &image, int bins = 16) {
    vector<float> histogram(bins * bins, 0.0);
    
    int totalPixels = 0;
    
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            Vec3b pixel = image.at<Vec3b>(i, j);
            
            float b = pixel[0];
            float g = pixel[1];
            float r = pixel[2];
            
            // Compute intensity
            float intensity = r + g + b;
            
            // Skip very dark pixels to avoid division by zero
            if(intensity < 1.0) {
                continue;
            }
            
            // Compute rg chromaticity
            float r_chrom = r / intensity;
            float g_chrom = g / intensity;
            
            // Map to bin indices [0, bins-1]
            int r_bin = (int)(r_chrom * bins);
            int g_bin = (int)(g_chrom * bins);
            
            // Clamp to valid range
            if(r_bin >= bins) r_bin = bins - 1;
            if(g_bin >= bins) g_bin = bins - 1;
            
            // Increment histogram bin
            int binIndex = r_bin * bins + g_bin;
            histogram[binIndex]++;
            totalPixels++;
        }
    }
    
    // Normalize histogram
    for(int i = 0; i < histogram.size(); i++) {
        histogram[i] = histogram[i] / totalPixels;
    }
    
    return histogram;
}

// Compute histogram intersection distance
// Returns distance (1 - intersection) so smaller is better
float histogramIntersection(vector<float> &hist1, vector<float> &hist2) {
    if(hist1.size() != hist2.size()) {
        printf("Error: Histograms have different sizes!\n");
        return -1.0;
    }
    
    float intersection = 0.0;
    
    for(int i = 0; i < hist1.size(); i++) {
        intersection += min(hist1[i], hist2[i]);
    }
    
    // Return distance (1 - intersection)
    // Higher intersection = more similar = smaller distance
    return 1.0 - intersection;
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
        printf("Example: %s images/pic.0164.jpg images 5\n", argv[0]);
        return -1;
    }
    
    char *targetImagePath = argv[1];
    char *imageDir = argv[2];
    int numMatches = atoi(argv[3]);
    int bins = 16; // 16x16 bins for rg chromaticity
    
    // Load target image
    Mat targetImage = imread(targetImagePath);
    if(targetImage.empty()) {
        printf("Error: Could not load target image: %s\n", targetImagePath);
        return -1;
    }
    
    printf("Target image: %s (%d x %d)\n", targetImagePath, targetImage.cols, targetImage.rows);
    printf("Using %dx%d rg chromaticity histogram\n", bins, bins);
    
    // Extract histogram from target image
    vector<float> targetHist = computeRGHistogram(targetImage, bins);
    printf("Computed histogram with %lu bins\n", targetHist.size());
    
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
            
            // Compute histogram
            vector<float> hist = computeRGHistogram(image, bins);
            
            // Compute distance using histogram intersection
            float distance = histogramIntersection(targetHist, hist);
            
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
    printf("\n=== Top %d matches ===\n", numMatches);
    for(int i = 0; i < min(numMatches, (int)matches.size()); i++) {
        printf("%d. %s (distance: %.4f)\n", i+1, matches[i].filename.c_str(), matches[i].distance);
    }
    
    return 0;
}