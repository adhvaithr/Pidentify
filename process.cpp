#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <thread>

#include "classMember.h"
#include "modelState.h"

void normalizeFeatures(std::vector<ClassMember>& dataset) {
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
            std::exit(0);
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
            std::exit(0);
        }
    }

    for (auto& obj : dataset) {
        for (size_t i = 0; i < dataset[0].features.size(); ++i) {
            obj.features[i] = (obj.features[i] - means[i]) / sigmas[i];
        }
    }

    // Save means and standard deviations used for standardization
    MODEL_STATE.means = std::move(means);
    MODEL_STATE.sigmas = std::move(sigmas);

}

double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

void computeNearestNeighborDistances(const std::unordered_map<std::string, std::vector<ClassMember> >& classMap,
    std::unordered_map<std::string, std::vector<double> >& classNNDistMap, std::string className) {
    for (const auto& obj : classMap.at(className)) {
        double minDistance = std::numeric_limits<double>::max();
        for (const auto& neighbor : classMap.at(className)) {
            if (&obj != &neighbor) {
                double distance = euclideanDistance(obj.features, neighbor.features);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
        }
        // If the nearest neighbor distance is greater than 1, drop the datapoint
        if (minDistance <= 1) {
            m.lock();
            classNNDistMap[className].push_back(minDistance);
            m.unlock();
        }
    }
}

std::unordered_map<std::string, std::vector<double> > process(std::vector<ClassMember> dataset) {

    // normalize features
    normalizeFeatures(dataset);

    // Group dataset by class
    std::unordered_map<std::string, std::vector<ClassMember> > classMap;
    for (const auto& obj : dataset) {
        classMap[obj.name].push_back(obj);
    }

    // compute k nearest distance, k = 1
    std::unordered_map<std::string, std::vector<double> > classNNDistMap;
    std::vector<std::thread> threads;
    
    for (const auto& pair : classMap) {
        std::thread t(computeNearestNeighborDistances, std::cref(classMap), std::ref(classNNDistMap), pair.first);
        threads.push_back(std::move(t));
    }

    for (auto& t : threads) {
        t.join();
    }

    // Check if a class has no datapoints that are within a distance of 1 from each other
    std::vector<std::string> invalidClasses;
    for (const auto& pair : classMap) {
        if (classNNDistMap.find(pair.first) == classNNDistMap.end()) {
            invalidClasses.push_back(pair.first);
        }
    }

    if (invalidClasses.size() > 0) {
        std::cout << "Unable to perform curve fitting for all classes.\n";
        std::cout << "Nearest neighbor distances are greater than 1 for these classes: ";
        std::copy(invalidClasses.begin(), invalidClasses.end(), std::ostream_iterator<std::string>(std::cout, ", "));
        std::cout << std::endl;
        std::exit(0);
    }

    // Save all datapoints for each class
    MODEL_STATE.classMap = std::move(classMap);

    // sort distances in ascending order
    for (auto& pair : classNNDistMap) {
        std::sort(pair.second.begin(), pair.second.end());

        // eliminate duplicated results
        pair.second.erase(unique(pair.second.begin(), pair.second.end()), pair.second.end());
    }

    return classNNDistMap;
}