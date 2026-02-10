/*
  Deep Network Embedding Matching using ResNet18 features
  
  Usage: deep_embedding_match <target_image> <csv_file> <num_matches>
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include "csv_util.h"

using namespace cv;
using namespace std;

// Compute L2 norm (Euclidean length) of a vector
float computeL2Norm(vector<float> &vec) {
    float sum = 0.0;
    for(int i = 0; i < vec.size(); i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Normalize vector by its L2 norm
vector<float> normalizeVector(vector<float> &vec) {
    vector<float> normalized;
    float norm = computeL2Norm(vec);
    
    if(norm > 0) {
        for(int i = 0; i < vec.size(); i++) {
            normalized.push_back(vec[i] / norm);
        }
    } else {
        normalized = vec;
    }
    
    return normalized;
}

// Compute dot product of two vectors
float dotProduct(vector<float> &vec1, vector<float> &vec2) {
    if(vec1.size() != vec2.size()) {
        printf("Error: Vectors have different sizes!\n");
        return 0.0;
    }
    
    float sum = 0.0;
    for(int i = 0; i < vec1.size(); i++) {
        sum += vec1[i] * vec2[i];
    }
    
    return sum;
}

// Compute cosine distance: d = 1 - cos(theta)
// where cos(theta) = dot product of normalized vectors
float cosineDistance(vector<float> &vec1, vector<float> &vec2) {
    // Normalize both vectors
    vector<float> norm1 = normalizeVector(vec1);
    vector<float> norm2 = normalizeVector(vec2);
    
    // Compute dot product (which is cos(theta) for normalized vectors)
    float cosTheta = dotProduct(norm1, norm2);
    
    // Clamp to [-1, 1] to avoid numerical issues
    if(cosTheta > 1.0) cosTheta = 1.0;
    if(cosTheta < -1.0) cosTheta = -1.0;
    
    // Return distance
    return 1.0 - cosTheta;
}

// Compute sum of squared differences
float sumSquaredDistance(vector<float> &vec1, vector<float> &vec2) {
    if(vec1.size() != vec2.size()) {
        printf("Error: Vectors have different sizes!\n");
        return -1.0;
    }
    
    float ssd = 0.0;
    for(int i = 0; i < vec1.size(); i++) {
        float diff = vec1[i] - vec2[i];
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
        printf("Usage: %s <target_image_name> <csv_file> <num_matches>\n", argv[0]);
        printf("Example: %s pic.0893.jpg data/ResNet18_olym.csv 5\n", argv[0]);
        return -1;
    }
    
    char *targetImageName = argv[1];
    char *csvFile = argv[2];
    int numMatches = atoi(argv[3]);
    
    printf("Target image: %s\n", targetImageName);
    printf("Loading embeddings from: %s\n", csvFile);
    
    // Read CSV file with embeddings
    vector<char *> filenames;
    vector<vector<float>> embeddings;
    
    if(read_image_data_csv(csvFile, filenames, embeddings, 0) != 0) {
        printf("Error: Failed to read CSV file\n");
        return -1;
    }
    
    printf("Loaded %lu embeddings, each with %lu dimensions\n", 
           embeddings.size(), embeddings[0].size());
    
    // Find target image in the database
    int targetIndex = -1;
    for(int i = 0; i < filenames.size(); i++) {
        if(strcmp(filenames[i], targetImageName) == 0) {
            targetIndex = i;
            break;
        }
    }
    
    if(targetIndex == -1) {
        printf("Error: Target image %s not found in CSV file\n", targetImageName);
        return -1;
    }
    
    printf("Found target image at index %d\n", targetIndex);
    
    vector<float> &targetEmbedding = embeddings[targetIndex];
    
    // Compute distances to all images
    vector<ImageMatch> matches;
    
    for(int i = 0; i < filenames.size(); i++) {
        // Compute cosine distance
        float distance = cosineDistance(targetEmbedding, embeddings[i]);
        
        // Alternative: use sum-squared distance
        // float distance = sumSquaredDistance(targetEmbedding, embeddings[i]);
        
        ImageMatch match;
        match.filename = string(filenames[i]);
        match.distance = distance;
        matches.push_back(match);
    }
    
    // Sort matches by distance (ascending)
    sort(matches.begin(), matches.end());
    
    // Display top N matches
    printf("\n=== Top %d matches (Deep Network Embeddings - Cosine Distance) ===\n", numMatches);
    for(int i = 0; i < min(numMatches, (int)matches.size()); i++) {
        printf("%d. %s (distance: %.4f)\n", i+1, matches[i].filename.c_str(), matches[i].distance);
    }
    
    // Clean up
    for(int i = 0; i < filenames.size(); i++) {
        delete[] filenames[i];
    }
    
    return 0;
}