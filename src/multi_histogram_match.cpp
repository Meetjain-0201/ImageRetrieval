/*
  Multi-Histogram Matching using top and bottom halves RGB histograms
  
  Usage: multi_histogram_match <target_image> <image_directory> <num_matches>
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

// Compute 3D RGB histogram for a region of the image
vector<float> computeRGBHistogram(Mat &image, int startRow, int endRow, int bins = 8) {
    vector<float> histogram(bins * bins * bins, 0.0);
    
    int totalPixels = 0;
    
    for(int i = startRow; i < endRow; i++) {
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

// Compute two histograms: top half and bottom half
pair<vector<float>, vector<float>> computeTopBottomHistograms(Mat &image, int bins = 8) {
    int midRow = image.rows / 2;
    
    vector<float> topHist = computeRGBHistogram(image, 0, midRow, bins);
    vector<float> bottomHist = computeRGBHistogram(image, midRow, image.rows, bins);
    
    return make_pair(topHist, bottomHist);
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

// Compute combined distance using two histograms with equal weighting
float computeMultiHistogramDistance(pair<vector<float>, vector<float>> &hist1, 
                                     pair<vector<float>, vector<float>> &hist2) {
    // Compute intersection for top halves
    float topIntersection = histogramIntersection(hist1.first, hist2.first);
    
    // Compute intersection for bottom halves
    float bottomIntersection = histogramIntersection(hist1.second, hist2.second);
    
    // Equal weighting: average the two intersections
    float avgIntersection = (topIntersection + bottomIntersection) / 2.0;
    
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
        printf("Example: %s images/pic.0274.jpg images 5\n", argv[0]);
        return -1;
    }
    
    char *targetImagePath = argv[1];
    char *imageDir = argv[2];
    int numMatches = atoi(argv[3]);
    int bins = 8; // 8x8x8 bins for RGB
    
    // Load target image
    Mat targetImage = imread(targetImagePath);
    if(targetImage.empty()) {
        printf("Error: Could not load target image: %s\n", targetImagePath);
        return -1;
    }
    
    printf("Target image: %s (%d x %d)\n", targetImagePath, targetImage.cols, targetImage.rows);
    printf("Using %dx%dx%d RGB histogram for top and bottom halves\n", bins, bins, bins);
    
    // Extract histograms from target image
    auto targetHists = computeTopBottomHistograms(targetImage, bins);
    printf("Computed top histogram: %lu bins\n", targetHists.first.size());
    printf("Computed bottom histogram: %lu bins\n", targetHists.second.size());
    
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
            
            // Compute histograms
            auto hists = computeTopBottomHistograms(image, bins);
            
            // Compute distance
            float distance = computeMultiHistogramDistance(targetHists, hists);
            
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