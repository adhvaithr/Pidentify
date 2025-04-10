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

// Delete elements from a vector assuming indices is sorted in ascending order
void deleteVectorElements(const std::vector<size_t>& indices, std::vector<double>& vec) {
    for (int i = indices.size() - 1; i >= 0; --i) {
        vec.erase(vec.begin() + indices[i]);
    }
}

// Delete certain features from the dataset assuming indices is sorted in ascending order
void removeFeatures(const std::vector<size_t>& indices, std::vector<ClassMember>& dataset) {
    for (auto& obj : dataset) {
        deleteVectorElements(indices, obj.features);
    }
}

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
    
    std::vector<size_t> zeroStdDeviation;
    for (double& sigma : sigmas) {
        sigma = std::sqrt(sigma / dataset.size());
        if (sigma == 0) {
            size_t idx = &sigma - &sigmas[0];
            std::cout << "Standard deviation is zero for feature index " << idx << std::endl;
            std::cout << "Removing feature with index " << idx << " for all instances.\n";
            zeroStdDeviation.push_back(idx);
        }
    }

    if (!zeroStdDeviation.empty()) {
        removeFeatures(zeroStdDeviation, dataset);
        deleteVectorElements(zeroStdDeviation, means);
        deleteVectorElements(zeroStdDeviation, sigmas);
        MODEL_STATE.zeroStdDeviation = std::move(zeroStdDeviation);
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

std::unordered_map<std::string, std::vector<double> > process(std::vector<ClassMember> dataset) {

    // normalize features
    normalizeFeatures(dataset);

    reduceDimensionality(dataset);

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