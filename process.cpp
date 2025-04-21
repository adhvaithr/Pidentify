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
#include <dataanalysis.h>
#include <linalg.h>

#include "classMember.h"
#include "modelState.h"

void normalizeFeatures(std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
    if (dataset.empty()) {
        std::cerr << "Dataset is empty!" << std::endl;
        return;
    }

    size_t numFeatures = dataset.begin()->second[0].features.size(), totalDatapoints = 0;
    std::vector<double> featureMeans(numFeatures, 0.0);
    std::unordered_map<std::string, std::vector<double> > classCentroids;

    // Calculate mean for each feature and mean vector for each class
    for (const auto& pair : dataset) {
        classCentroids[pair.first].resize(numFeatures, 0.0);
        for (const auto& obj : pair.second) {
            if (obj.features.size() != numFeatures) {
                fprintf(stderr, "Inconsistent feature size: %zu != %zu\n", obj.features.size(), numFeatures);
                std::exit(0);
            }
            for (size_t i = 0; i < numFeatures; ++i) {
                featureMeans[i] += obj.features[i];
                classCentroids[pair.first][i] += obj.features[i];
            }
        }
        totalDatapoints += pair.second.size();
        for (size_t i = 0; i < numFeatures; ++i) {
            classCentroids[pair.first][i] /= pair.second.size();
        }
    }

    for (double& mean : featureMeans) {
        mean /= totalDatapoints;
    }

    // Calculate the minimum radius (smallest standard deviation of distance from the center of the class) across all classes
    double minRadius = std::numeric_limits<double>::max();
    for (const auto& pair : dataset) {
        std::vector<double> sigmas(numFeatures, 0.0);
        double radius = 0;
        for (const auto& obj : pair.second) {
            for (size_t i = 0; i < numFeatures; ++i) {
                sigmas[i] += (obj.features[i] - classCentroids[pair.first][i]) *
                    (obj.features[i] - classCentroids[pair.first][i]);
            }
        }
        for (double& sigma : sigmas) {
            if (sigma == 0) {
                std::cout << "Warning: For class \"" << pair.first << "\" the standard deviation of distance from the " <<
                    "centroid is 0 for feature index " << (&sigma - &sigmas[0]) << std::endl;
            }
            radius += sigma / pair.second.size();
        }
        radius = std::sqrt(radius);

        if (radius > 0 && radius < minRadius) {
            minRadius = radius;
        }
    }

    if (minRadius == std::numeric_limits<double>::max()) {
        std::cerr << "ERROR: Minimum radius is infinity\n";
        std::exit(0);
    }

    // Standardize datapoints
    for (auto& pair : dataset) {
        for (ClassMember& obj : pair.second) {
            for (size_t i = 0; i < numFeatures; ++i) {
               obj.features[i] = (obj.features[i] - featureMeans[i]) / minRadius;
            }
        }
    }

    MODEL_STATE.featureMeans = std::move(featureMeans);
    MODEL_STATE.minRadius = minRadius;
}

void copyDatapoints(std::vector<ClassMember>& dataset, alglib::real_2d_array& datapoints, bool to_alglib_array) {
    size_t npoints = dataset.size();
    size_t nvars = dataset[0].features.size();
    size_t projectionDim = datapoints.cols();

    if (to_alglib_array) {
        for (size_t i = 0; i < npoints; ++i) {
            for (size_t j = 0; j < nvars; ++j) {
                datapoints[i][j] = dataset[i].features[j];
            }
        }
    }
    else {
        for (size_t i = 0; i < npoints; ++i) {
            for (size_t j = 0; j < projectionDim; ++j) {
                dataset[i].features[j] = datapoints[i][j];
            }
            dataset[i].features.resize(projectionDim);
        }
    }
}

/* Project a matrix of datapoints, where each row corresponds to a different datapoint, onto the principal axes,
where each column is a vector of the basis, and store the results in principalComponents. */
void projectOntoPrincipalAxes(const alglib::real_2d_array& datapoints, const alglib::real_2d_array& principalAxes,
    alglib::real_2d_array& principalComponents) {
    alglib::rmatrixgemm(datapoints.rows(), principalAxes.cols(), datapoints.cols(), 1, datapoints, 0, 0, 0, principalAxes, 0, 0, 0, 0, principalComponents, 0, 0);
}

// Project a higher dimension dataset into a lower dimension subspace
void reduceDimensionality(std::vector<ClassMember>& dataset) {
    alglib::real_2d_array datapoints;
    alglib::real_1d_array variance;
    alglib::real_2d_array principalAxes;
    alglib::real_2d_array principalComponents;
    size_t npoints = dataset.size();
    size_t nvars = dataset[0].features.size();
    alglib::ae_int_t nneeded = 3, eps = 0, maxits = 0;

    datapoints.setlength(npoints, nvars);
    variance.setlength(nneeded);
    principalAxes.setlength(nvars, nneeded);
    principalComponents.setlength(npoints, nneeded);

    // Copy data into ALGLIB array
    copyDatapoints(dataset, datapoints, true);

    // Find the principal axes
    alglib::pcatruncatedsubspace(datapoints, nneeded, eps, maxits, variance, principalAxes);

    // Project dataset into lower dimension
    projectOntoPrincipalAxes(datapoints, principalAxes, principalComponents);

    // Copy from ALGLIB array into vector of ClassMember objects
    copyDatapoints(dataset, principalComponents, false);

    // Save principal axes for reducing dimensionality of test set
    MODEL_STATE.principalAxes = std::move(principalAxes);
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

        // Record neirest neighbor distance
        m.lock();
        classNNDistMap[className].push_back(minDistance);
        m.unlock();
        
    }
}

std::unordered_map<std::string, std::vector<ClassMember> > groupByClass(std::vector<ClassMember>& dataset) {
    std::unordered_map<std::string, std::vector<ClassMember> > classMap;
    for (auto& obj : dataset) {
        classMap[obj.name].push_back(obj);
    }
    return classMap;
}

std::unordered_map<std::string, std::vector<double> > process(std::vector<ClassMember> dataset, bool applyPCA) {
    std::unordered_map<std::string, std::vector<ClassMember> > classMap = groupByClass(dataset);

    // normalize features
    normalizeFeatures(classMap);
    
    if (applyPCA && dataset[0].features.size() >= 3) {
        size_t i = 0;
        for (const auto& pair : classMap) {
            for (const ClassMember& datapoint : pair.second) {
                dataset[i++] = datapoint;
            }
        }

        reduceDimensionality(dataset);

        classMap = groupByClass(dataset);
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

    std::vector<std::string> warningClasses;

    for (const auto& pair : classMap) {
        const auto& currClass = pair.first; 
        auto distance_it = classNNDistMap.find(currClass);
        const std::vector<double>& distances = distance_it->second;
        double minDistance = *std::min_element(distances.begin(), distances.end());
        if (minDistance > 1.0) {
            warningClasses.push_back(currClass);
        }
    }
    if (warningClasses.size() > 0) {
        std::cout << "Nearest neighbor distances are greater than 1 for these classes: ";
        std::copy(warningClasses.begin(), warningClasses.end(), std::ostream_iterator<std::string>(std::cout, ", "));
        std::cout << std::endl;

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