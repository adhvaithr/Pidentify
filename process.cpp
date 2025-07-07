#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <dataanalysis.h>
#include <linalg.h>
#include "nanoflann/nanoflann.hpp"
#include "nanoflann/KDTreeVectorOfVectorsAdaptor.h"

#include "classMember.h"
#include "modelState.h"

// Delete elements from a vector assuming indices is sorted in ascending order
void deleteVectorElements(const std::vector<size_t>& indices, std::vector<double>& vec) {
    for (int i = indices.size() - 1; i >= 0; --i) {
        vec.erase(vec.begin() + indices[i]);
    }
}

// Delete certain features from the dataset assuming indices is sorted in ascending order
void removeFeatures(const std::vector<size_t>& indices, std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
    for (auto& pair : dataset) {
        for (auto& obj : pair.second) {
            deleteVectorElements(indices, obj.features);
        }
    }
}

// Delete certain features from the dataset assuming indices is sorted in ascending order
void removeFeatures(const std::vector<size_t>& indices, std::vector<ClassMember>& dataset) {
    for (auto& obj : dataset) {
        deleteVectorElements(indices, obj.features);
    }
}

void standardizeFeatures(std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
    if (dataset.empty()) {
        std::cerr << "Dataset is empty!" << std::endl;
        return;
    }

    size_t numFeatures = dataset.begin()->second[0].features.size(), totalDatapoints = 0;
    std::vector<double> means(numFeatures, 0.0);
    std::vector<double> sigmas(numFeatures, 0.0);

    // Calculate the mean for each feature
    for (const auto& pair : dataset) {
        totalDatapoints += pair.second.size();
        for (const auto& obj : pair.second) {
            if (obj.features.size() != numFeatures) {
                fprintf(stderr, "Inconsistent feature size: %zu != %zu\n", obj.features.size(), numFeatures);
                std::exit(0);
            }
            for (size_t i = 0; i < numFeatures; ++i) {
                means[i] += obj.features[i];
            }
        }
    }

    for (double& mean : means) {
        mean /= totalDatapoints;
    }

    MODEL_STATE.trainDatasetSize = totalDatapoints;

    // Calculate the standard deviation for each feature
    for (const auto& pair : dataset) {
        for (const auto& obj : pair.second) {
            for (size_t i = 0; i < numFeatures; ++i) {
                sigmas[i] += (obj.features[i] - means[i]) * (obj.features[i] - means[i]);
            }
        }
    }

    std::vector<size_t> zeroStdDeviation;
    for (double& sigma : sigmas) {
        sigma = std::sqrt(sigma / totalDatapoints);
        if (sigma == 0) {
            size_t idx = &sigma - &sigmas[0];
            std::cout << "Standard deviation is zero for feature index " << idx << std::endl;
            std::cout << "Removing feature with index " << idx << " for all instances.\n";
            zeroStdDeviation.push_back(idx);
        }
    }

    // Remove any features that have a standard deviation of 0
    if (!zeroStdDeviation.empty()) {
        removeFeatures(zeroStdDeviation, dataset);
        deleteVectorElements(zeroStdDeviation, means);
        deleteVectorElements(zeroStdDeviation, sigmas);
        MODEL_STATE.zeroStdDeviation = std::move(zeroStdDeviation);
    }

    // Z-score standardization of each feature
    for (auto& pair : dataset) {
        for (auto& obj : pair.second) {
            for (size_t i = 0; i < obj.features.size(); ++i) {
                obj.features[i] = (obj.features[i] - means[i]) / sigmas[i];
            }
        }
    }

    // Save means and standard deviations to standardize test points
    MODEL_STATE.means = std::move(means);
    MODEL_STATE.sigmas = std::move(sigmas);
}

void copyDatapoints(std::unordered_map<std::string, std::vector<ClassMember> >& dataset, alglib::real_2d_array& datapoints,
    bool to_alglib_array) {
    if (to_alglib_array) {
        size_t i = 0;
        size_t nvars = dataset.begin()->second[0].features.size();
        for (const std::string& className : MODEL_STATE.classNames) {
            for (const auto& obj : dataset.at(className)) {
                for (size_t j = 0; j < nvars; ++j) {
                    datapoints[i][j] = obj.features[j];
                }
                ++i;
            }
        }
    }
    else {
        size_t projectionDim = datapoints.cols();
        size_t i = 0;
        for (const std::string& className : MODEL_STATE.classNames) {
            for (size_t k = 0; k < dataset.at(className).size(); ++k) {
                for (size_t j = 0; j < projectionDim; ++j) {
                    dataset.at(className)[k].features[j] = datapoints[i][j];
                }
                dataset.at(className)[k].features.resize(projectionDim);
                ++i;
            }
        }
    }
}

/* Project a matrix of datapoints, where each row corresponds to a different datapoint, onto the principal axes,
where each column is a vector of the basis, and store the results in principalComponents. */
void projectOntoPrincipalAxes(const alglib::real_2d_array& datapoints, const alglib::real_2d_array& principalAxes,
    alglib::real_2d_array& principalComponents) {
    alglib::rmatrixgemm(datapoints.rows(), principalAxes.cols(), datapoints.cols(), 1, datapoints, 0, 0, 0, principalAxes, 0, 0, 0, 0, principalComponents, 0, 0);
}

void reduceDimensionality(std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
    alglib::real_2d_array datapoints;
    alglib::real_1d_array variance;
    alglib::real_2d_array principalAxes;
    alglib::real_2d_array principalComponents;
    size_t npoints = MODEL_STATE.trainDatasetSize;
    size_t nvars = dataset.begin()->second[0].features.size();

    datapoints.setlength(npoints, nvars);
    variance.setlength(nvars);
    principalAxes.setlength(nvars, nvars);

    // Copy data into ALGLIB array
    copyDatapoints(dataset, datapoints, true);

    // Find the principal axes
    alglib::pcabuildbasis(datapoints, variance, principalAxes);

    size_t basisDimension = 0;

    for (size_t i = 0; variance[i++] > 0.1; ++basisDimension) {}

    alglib::real_2d_array newBasis;
    newBasis.setlength(nvars, basisDimension);
    principalComponents.setlength(npoints, basisDimension);
    for (size_t i = 0; i < nvars; ++i) {
        for (size_t j = 0; j < basisDimension; ++j) {
            newBasis[i][j] = principalAxes[i][j];
        }
    }

    // Project dataset into lower dimension
    projectOntoPrincipalAxes(datapoints, newBasis, principalComponents);

    // Copy from ALGLIB array into vector of ClassMember objects
    copyDatapoints(dataset, principalComponents, false);

    // Save principal axes for reducing dimensionality of test set
    MODEL_STATE.principalAxes.setlength(nvars, basisDimension);
    MODEL_STATE.principalAxes = std::move(newBasis);
}

double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

double weightedEuclideanDistance(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& weights) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += weights[i] * (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// Return dataset that only contains datapoints with at least l out of k nearest neighbors
// of the same class using euclidean distance
std::unordered_map<std::string, std::vector<std::vector<double> > > filterDatapoints(
    std::unordered_map<std::string, std::vector<ClassMember> >& classMap, int k, int l) {    
    std::unordered_map<std::string, std::vector<std::vector<double> > > filteredDataset;
    std::vector<std::vector<double> > dataset(MODEL_STATE.trainDatasetSize);

    // Create matrix, i.e. vector of vectors, of all points in training dataset
    size_t i = 0;
    for (const std::string& className : MODEL_STATE.classNames) {
        for (const auto& obj : classMap.at(className)) {
            dataset[i].insert(dataset[i].end(), obj.features.begin(), obj.features.end());
            ++i;
        }
    }

    size_t dim = classMap.begin()->second[0].features.size();
    KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double> matIndex(dim, dataset, 10, 0);

    auto classNameIter = MODEL_STATE.classNames.begin();
    size_t classStart = 0, classEnd = classMap.at(*classNameIter).size();
    for (size_t i = 0; i < MODEL_STATE.trainDatasetSize; ++i) {
        if (i >= classEnd) {
            classStart = classEnd;
            classEnd += classMap.at(*(++classNameIter)).size();
        }
        std::vector<size_t> neighborIndices(k + 1);
        std::vector<double> squaredDistances(k + 1);

        size_t neighborsFound = matIndex.index->knnSearch(&dataset[i][0], k + 1, &neighborIndices[0], &squaredDistances[0]);
        if (neighborsFound < k + 1) {
            std::cout << "Can not find k = " << k << " neighbors due to not enough points in training set\n";
            std::exit(0);
        }

        size_t sameClassNeighbors = 0;
        std::for_each(neighborIndices.begin() + 1, neighborIndices.end(),
            [&sameClassNeighbors, classStart, classEnd](size_t i) {
                if (i >= classStart && i < classEnd) {
                    ++sameClassNeighbors;
                }
            });

        // Only keep datapoints that have l or more nearest neighbors of the same class out of k nearest neighbors
        if (sameClassNeighbors >= l) {
            filteredDataset[*classNameIter].push_back(dataset[i]);
        }        
    }

    // Check if there are only 0 or 1 classes remaining after filtering
    if (filteredDataset.size() <= 1) {
        std::cout << "Filtering of datapoints with l=" << l << " and k=" << k
            << " results in " << filteredDataset.size() << " remaining classes: ";
        if (filteredDataset.size() == 1) {
            std::cout << filteredDataset.begin()->first << std::endl;
        }
        std::exit(0);
    }

    return filteredDataset;
}

std::unordered_map<std::string, std::vector<double> > computeNearestNeighborDistances(
    const std::unordered_map<std::string, std::vector<std::vector<double> > >& classMap) {
    std::unordered_map<std::string, std::vector<double> > classNNDistMap;
    size_t dim = classMap.begin()->second[0].size();
    size_t k = 1;
    for (const auto& pair : classMap) {        
        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double> matIndex(dim, pair.second, 10, 0);

        for (const std::vector<double>& datapoint : pair.second) {
            std::vector<size_t> neighborIndices(k + 1);
            std::vector<double> squaredDistances(k + 1);

            nanoflann::KNNResultSet<double> resultSet(k + 1);
            resultSet.init(&neighborIndices[0], &squaredDistances[0]);
            matIndex.index->findNeighbors(resultSet, &datapoint[0]);

            classNNDistMap[pair.first].push_back(std::sqrt(squaredDistances[k]));
        }        
    }

    return classNNDistMap;
}

void weightFeatures(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
    size_t numFeatures = dataset.begin()->second[0].features.size();

    for (const auto& pair : dataset) {
        std::vector<double> MWCFD(numFeatures, 0.0); // Mean Within Class Feature Distances
        std::vector<double> MOCFD(numFeatures, 0.0); // Mean Outside Class Feature Distances
        const std::vector<double> *nearestWithinClass = nullptr, *nearestOutsideClass = nullptr;

        for (const auto& obj : pair.second) {
            // Find the closest datapoint within the same class
            double minDistance = std::numeric_limits<double>::max();
            for (const auto& neighbor : dataset.at(pair.first)) {
                if (&obj != &neighbor) {
                    double distance = euclideanDistance(obj.features, neighbor.features);
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestWithinClass = &neighbor.features;
                    }
                }
            }

            // Find the closest datapoint outside the class
            minDistance = std::numeric_limits<double>::max();
            for (const std::string className : MODEL_STATE.classNames) {
                if (className != pair.first) {
                    for (const auto& outsideNeighbor : dataset.at(className)) {
                        double distance = euclideanDistance(obj.features, outsideNeighbor.features);
                        if (distance < minDistance) {
                            minDistance = distance;
                            nearestOutsideClass = &outsideNeighbor.features;
                        }
                    }
                }
            }

            std::vector<double> featureDistances(numFeatures);
            std::transform(obj.features.begin(), obj.features.end(), nearestWithinClass->begin(), featureDistances.begin(),
                [](double feature1, double feature2)
                {return (feature1 - feature2 < 0) ? feature2 - feature1 : feature1 - feature2; });

            std::transform(MWCFD.begin(), MWCFD.end(), featureDistances.begin(), MWCFD.begin(),
                [](double cumDistance, double distance) {return cumDistance + distance; });

            std::transform(obj.features.begin(), obj.features.end(), nearestOutsideClass->begin(), featureDistances.begin(),
                [](double feature1, double feature2)
                {return (feature1 - feature2 < 0) ? feature2 - feature1 : feature1 - feature2; });

            std::transform(MOCFD.begin(), MOCFD.end(), featureDistances.begin(), MOCFD.begin(),
                [](double cumDistance, double distance) {return cumDistance + distance; });
        }

        size_t numClassInstances = pair.second.size();
        std::transform(MWCFD.begin(), MWCFD.end(), MWCFD.begin(), [numClassInstances](double cumDistance) {return cumDistance / numClassInstances; });
        std::transform(MOCFD.begin(), MOCFD.end(), MOCFD.begin(), [numClassInstances](double cumDistance) {return cumDistance / numClassInstances; });

        if (MODEL_STATE.featureWeights[pair.first].size() != numFeatures) {
            MODEL_STATE.featureWeights[pair.first].resize(numFeatures);
        }

        std::transform(MOCFD.begin(), MOCFD.end(), MWCFD.begin(), MODEL_STATE.featureWeights[pair.first].begin(),
            [](double outsideMeanDistance, double withinMeanDistance)
            {return std::max((outsideMeanDistance / withinMeanDistance) - 1.0, 0.0); });
    }
}

std::unordered_map<std::string, std::vector<double> > process(std::unordered_map<std::string,
    std::vector<ClassMember> >& dataset, int numNeighborsChecked, int minSameClassCount) {
   // Z-score standardization of features
   standardizeFeatures(dataset);

   if (MODEL_STATE.processType == "PCA") {
       reduceDimensionality(dataset);
   }
   else if (MODEL_STATE.processType == "featureWeighting") {
       weightFeatures(dataset);
   }

   std::unordered_map<std::string, std::vector<std::vector<double> > > filteredDataset = filterDatapoints(dataset,
        numNeighborsChecked, minSameClassCount);
   
   // compute k nearest distance, k = 1
   std::unordered_map<std::string, std::vector<double> > classNNDistMap = computeNearestNeighborDistances(filteredDataset);

   // Save all datapoints for each class
   MODEL_STATE.classMap = std::move(filteredDataset);

   std::vector<std::string> warningClasses;

   // sort distances in ascending order
   for (auto& pair : classNNDistMap) {
       std::sort(pair.second.begin(), pair.second.end());

       // eliminate duplicated results
       pair.second.erase(unique(pair.second.begin(), pair.second.end()), pair.second.end());

       if (pair.second[0] > 1.0) {
           warningClasses.push_back(pair.first);
       }
   }

   if (warningClasses.size() > 0) {
       std::cout << "Nearest neighbor distances are greater than 1 for these classes: ";
       std::copy(warningClasses.begin(), warningClasses.end(), std::ostream_iterator<std::string>(std::cout, ", "));
       std::cout << std::endl;
   }

   return classNNDistMap;
}
