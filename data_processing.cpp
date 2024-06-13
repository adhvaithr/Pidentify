#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <limits>
#include <assert.h>
#include <unordered_map>
#include <algorithm>
#include "iris.h"

void normalizeFeatures(std::vector<Iris>& dataset) {
    if (dataset.empty()) {
        std::cerr << "Dataset is empty!" << std::endl;
        return;
    }

    size_t numFeatures = dataset[0].features.size();
    std::vector<double> means(numFeatures, 0.0);
    std::vector<double> sigmas(numFeatures, 0.0);

    // Calculate mean for each feature
    for (const auto& obj : dataset) {
        if (obj.features.size() != numFeatures) {
            fprintf(stderr, "Inconsistent feature size: %zu != %zu\n", obj.features.size(), numFeatures);
            return;
        }
        for (size_t i = 0; i < numFeatures; ++i) {
            means[i] += obj.features[i];
        }
    }

    for (double& mean : means) {
        mean /= dataset.size();
    }

    // Calculate standard deviation for each feature
    for (const auto& obj : dataset) {
        for (size_t i = 0; i < numFeatures; ++i) {
            sigmas[i] += (obj.features[i] - means[i]) * (obj.features[i] - means[i]);
        }
    }

    for (double& sigma : sigmas) {
        sigma = std::sqrt(sigma / dataset.size());
        if (sigma == 0) {
            std::cerr << "Standard deviation is zero for feature index " << (&sigma - &sigmas[0]) << std::endl;
            return;
        }
    }

    // Normalize the dataset
    for (auto& obj : dataset) {
        for (size_t i = 0; i < numFeatures; ++i) {
            obj.features[i] = (obj.features[i] - means[i]) / sigmas[i];
        }
    }
}


double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}


std::vector<double> computeNearestNeighborDistances(const std::vector<Iris>& dataset) {
    std::unordered_map<std::string, std::vector<Iris> > classMap;
    std::vector<double> distances;
    
    // Group dataset by species
    for (const auto& obj : dataset) {
        classMap[obj.species].push_back(obj);
    }
    
    // Compute nearest neighbor distances for each class
    for (const auto& pair : classMap) {
        const auto& classData = pair.second;

        for (const auto& obj : classData) {
            double minDistance = std::numeric_limits<double>::max();
            for (const auto& neighbor : classData) {
                if (&obj != &neighbor) {
                    double distance = euclideanDistance(obj.features, neighbor.features);
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
            }
            // if the result of distance is bigger than 1, it will be dropped.
            if (minDistance <= 1) {
                distances.push_back(minDistance);
            }
        }
    }

    return distances;
}

std::vector<double> process(std::vector<Iris> dataset){

    // normalize features
    normalizeFeatures(dataset);

    // computer k nearest distance, k = 1
    std::vector<double> distances = computeNearestNeighborDistances(dataset);

    // sort distances in ascending order
    std::sort(distances.begin(), distances.end());

    // eliminate duplicated results
    distances.erase(unique(distances.begin(), distances.end()),distances.end());

    return distances;
}