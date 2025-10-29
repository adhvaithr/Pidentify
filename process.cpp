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

#include "process.h"
#include "classMember.h"
#include "modelState.h"
#include "nanoflann/WeightedL2MetricAdaptor.hpp"

#include "CSVWrite.hpp"
#include <cstdlib>

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

    for (size_t i = 0; i < nvars; ++i) {
        if (variance[i] >= MIN_PCA_BASIS_VARIANCE) {
            ++basisDimension;
        }
    }

    if (basisDimension < MIN_PCA_BASIS) {
        printf("Warning: Insufficient dimension for projection to new basis when a minimum variance of "
            "%g is required.  Defaulting to %lu-D basis with highest variance possible.\n",
            MIN_PCA_BASIS_VARIANCE, MIN_PCA_BASIS);
        basisDimension = MIN_PCA_BASIS;
    }

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
    MODEL_STATE.principalAxes = newBasis;
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
    const std::unordered_map<std::string, std::vector<ClassMember> >& classMap, int k, int l) {    
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

    // Drop classes that don't have enough points after filtering
    std::vector<std::string> droppedClasses;
    for (const std::string& className : MODEL_STATE.classNames) {
        if (filteredDataset.find(className) == filteredDataset.end()) {
            droppedClasses.push_back(className);
        }
        else if (filteredDataset[className].size() < MIN_CLASS_MEMBERS) {
            droppedClasses.push_back(className);
            filteredDataset.erase(className);
        }
    }
    if (!droppedClasses.empty()) {
        std::cout << "Warning: Filtering of datapoints in the training set by l="
            << l << " and k=" << k << " results in an insufficient number of remaining "
            "points for classes: ";
        std::copy(droppedClasses.begin(), droppedClasses.end(), std::ostream_iterator<std::string>(std::cout, ", "));
        std::cout << std::endl;
        std::cout << "Classes have been removed from training set\n";
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

std::unordered_map<std::string, std::vector<std::vector<double> > > weightedFilterDatapoints(
    const std::unordered_map<std::string, std::vector<ClassMember> >& classMap, int k, int l) {
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
    std::vector<size_t> neighborIndices(k + 1);
    std::vector<double> squaredDistances(k + 1);
    size_t classStart = 0, classEnd = 0;

    for (const std::string& className : MODEL_STATE.classNames) {
        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double, -1,
            metric_Weighted_L2> matIndex(dim, dataset, 10, 0, MODEL_STATE.featureWeights.at(className));
        classStart = classEnd;
        classEnd += classMap.at(className).size();

        for (size_t i = classStart; i < classEnd; ++i) {
            size_t neighborsFound = matIndex.index->knnSearch(&dataset[i][0], k + 1, &neighborIndices[0], &squaredDistances[0]);
            if (neighborsFound < k + 1) {
                std::cout << "Can not find k = " << k << " neighbors due to not enough points in training set\n";
                std::exit(0);
            }

            size_t sameClassNeighbors = 0;
            std::for_each(neighborIndices.begin(), neighborIndices.end(),
                [&sameClassNeighbors, classStart, classEnd, i](size_t neighborIdx) {
                    if (neighborIdx >= classStart && neighborIdx < classEnd && neighborIdx != i) {
                        ++sameClassNeighbors;
                    }
                });

            // Only keep datapoints that have l or more nearest neighbors of the same class out of k nearest neighbors
            if (sameClassNeighbors >= l) {
                filteredDataset[className].push_back(dataset[i]);
            }
        }
    }
    
    // Drop classes that don't have enough points after filtering
    std::vector<std::string> droppedClasses;
    for (const std::string& className : MODEL_STATE.classNames) {
        if (filteredDataset.find(className) == filteredDataset.end()) {
            droppedClasses.push_back(className);
        }
        else if (filteredDataset[className].size() < MIN_CLASS_MEMBERS) {
            droppedClasses.push_back(className);
            filteredDataset.erase(className);
        }
    }
    if (!droppedClasses.empty()) {
        std::cout << "Warning: Filtering of datapoints in the training set by l="
            << l << " and k=" << k << " results in an insufficient number of remaining "
            "points for classes: ";
        std::copy(droppedClasses.begin(), droppedClasses.end(), std::ostream_iterator<std::string>(std::cout, ", "));
        std::cout << std::endl;
        std::cout << "Classes have been removed from training set\n";
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

void weightFeatures(const std::unordered_map<std::string, std::vector<ClassMember> >& classMap) {
    size_t dim = classMap.begin()->second[0].features.size();
    size_t k = 1;
    std::vector<std::vector<double> > masterDataset(MODEL_STATE.trainDatasetSize, std::vector<double>(dim));

    size_t i = 0;
    for (const std::string& className : MODEL_STATE.classNames) {
        for (const ClassMember& obj : classMap.at(className)) {
            std::copy(obj.features.begin(), obj.features.end(), masterDataset[i++].begin());
        }
    }

    std::vector<std::vector<double>>::iterator start, end = masterDataset.begin();
    for (const std::string& className : MODEL_STATE.classNames) {
        size_t classSize = classMap.at(className).size();
        start = end;
        end += classSize;

        // Construct matrices for all datapoints in the same class and all datapoints in different classes
        std::vector<std::vector<double>> WCDataset(start, end);
        std::vector<std::vector<double>> OCDataset(MODEL_STATE.trainDatasetSize - classSize, std::vector<double>(dim));
        if (start == masterDataset.begin()) {
            std::copy(end, masterDataset.end(), OCDataset.begin());
        }
        else if (end == masterDataset.end()) {
            std::copy(masterDataset.begin(), start, OCDataset.begin());
        }
        else {
            std::copy(masterDataset.begin(), start, OCDataset.begin());
            std::copy(end, masterDataset.end(), OCDataset.begin() + (start - masterDataset.begin()));
        }        

        std::vector<double> MWCFD(dim, 0.0); // Mean Within Class Feature Distances
        std::vector<double> MOCFD(dim, 0.0); // Mean Outside Class Feature Distances

        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double> WCTree(dim, WCDataset, 10, 0);
        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double> OCTree(dim, OCDataset, 10, 0);
        std::vector<size_t> WCNeighborIndices(k + 1);
        std::vector<double> WCSquaredDistances(k + 1);
        std::vector<size_t> OCNeighborIndices(k);
        std::vector<double> OCSquaredDistances(k);

        for (size_t i = 0; i < classSize; ++i) {
            // Add difference between vectors for current datapoint and closest datapoint within the same class to MWCFD
            WCTree.query(&WCDataset[i][0], k + 1, &WCNeighborIndices[0], &WCSquaredDistances[0]);
            size_t j = 0;
            std::transform(WCDataset[i].cbegin(), WCDataset[i].cend(), WCDataset[WCNeighborIndices[k]].cbegin(),
                MWCFD.begin(), [&MWCFD, &j](double feature1, double feature2) {return MWCFD[j++] + std::abs(feature1 - feature2); });

            // Add difference between vectors for current datapoint and closest datapoint outside the class to MOCFD
            OCTree.query(&WCDataset[i][0], k, &OCNeighborIndices[0], &OCSquaredDistances[0]);
            j = 0;
            std::transform(WCDataset[i].cbegin(), WCDataset[i].cend(), OCDataset[OCNeighborIndices[0]].cbegin(),
                MOCFD.begin(), [&MOCFD, &j](double feature1, double feature2) {return MOCFD[j++] + std::abs(feature1 - feature2); });
        }

        // Calculate feature weights for the class
        std::transform(MWCFD.begin(), MWCFD.end(), MWCFD.begin(), [classSize](double cumDistance) {return cumDistance / classSize; });
        std::transform(MOCFD.begin(), MOCFD.end(), MOCFD.begin(), [classSize](double cumDistance) {return cumDistance / classSize; });

        if (MODEL_STATE.featureWeights[className].size() != dim) {
            MODEL_STATE.featureWeights[className].resize(dim);
        }

        std::transform(MOCFD.cbegin(), MOCFD.cend(), MWCFD.cbegin(), MODEL_STATE.featureWeights[className].begin(),
            [&className](double outsideMeanDistance, double withinMeanDistance) {
                if (withinMeanDistance == 0) {
                    withinMeanDistance = 1.0 / MODEL_STATE.numInstancesPerClass.at(className);
                    /*
                    size_t totalDatasetSize = std::accumulate(MODEL_STATE.numInstancesPerClass.begin(), MODEL_STATE.numInstancesPerClass.end(), 0,
                        [](size_t a, const std::pair<std::string, double>& b) {return a + b.second; });
                    withinMeanDistance = 1.0 / totalDatasetSize;
                    */
                }
                return std::max((outsideMeanDistance / withinMeanDistance) - 1.0, 0.0);
            }
        );
    }
}

void weightFeatures(const std::unordered_map<std::string, std::vector<std::vector<double> > >& classMap) {
    size_t dim = classMap.begin()->second[0].size();
    size_t k = 1;

    for (const auto& currentClass : classMap) {
        // Construct matrix for all datapoints in different classes
        std::vector<std::vector<double>> OCDataset;
        for (const auto& classData : classMap) {
            if (classData.first != currentClass.first) {
                OCDataset.insert(OCDataset.end(), classData.second.begin(), classData.second.end());
            }
        }

        std::vector<double> MWCFD(dim, 0.0); // Mean Within Class Feature Distances
        std::vector<double> MOCFD(dim, 0.0); // Mean Outside Class Feature Distances

        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double> WCTree(dim, currentClass.second, 10, 0);
        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double> OCTree(dim, OCDataset, 10, 0);
        std::vector<size_t> WCNeighborIndices(k + 1);
        std::vector<double> WCSquaredDistances(k + 1);
        std::vector<size_t> OCNeighborIndices(k);
        std::vector<double> OCSquaredDistances(k);

        size_t classSize = currentClass.second.size();
        for (size_t i = 0; i < classSize; ++i) {
            // Add difference between vectors for current datapoint and closest datapoint within the same class to MWCFD
            WCTree.query(&currentClass.second[i][0], k + 1, &WCNeighborIndices[0], &WCSquaredDistances[0]);
            size_t j = 0;
            std::transform(currentClass.second[i].cbegin(), currentClass.second[i].cend(), currentClass.second[WCNeighborIndices[k]].cbegin(),
                MWCFD.begin(), [&MWCFD, &j](double feature1, double feature2) {return MWCFD[j++] + std::abs(feature1 - feature2); });

            // Add difference between vectors for current datapoint and closest datapoint outside the class to MOCFD
            OCTree.query(&currentClass.second[i][0], k, &OCNeighborIndices[0], &OCSquaredDistances[0]);
            j = 0;
            std::transform(currentClass.second[i].cbegin(), currentClass.second[i].cend(), OCDataset[OCNeighborIndices[0]].cbegin(),
                MOCFD.begin(), [&MOCFD, &j](double feature1, double feature2) {return MOCFD[j++] + std::abs(feature1 - feature2); });
        }

        // Calculate feature weights for the class
        std::transform(MWCFD.begin(), MWCFD.end(), MWCFD.begin(), [classSize](double cumDistance) {return cumDistance / classSize; });
        std::transform(MOCFD.begin(), MOCFD.end(), MOCFD.begin(), [classSize](double cumDistance) {return cumDistance / classSize; });

        if (MODEL_STATE.featureWeights[currentClass.first].size() != dim) {
            MODEL_STATE.featureWeights[currentClass.first].resize(dim);
        }
        // Assign a weight of 0 if withinMeanDistance = 0
        std::transform(MOCFD.cbegin(), MOCFD.cend(), MWCFD.cbegin(), MODEL_STATE.featureWeights[currentClass.first].begin(),
            [](double outsideMeanDistance, double withinMeanDistance)
            {return (withinMeanDistance == 0) ? 0 : std::max((outsideMeanDistance / withinMeanDistance) - 1.0, 0.0); });
    }
}

template <
    class VectorOfVectorsType, typename num_t, int DIM = -1,
    class Distance = nanoflann::metric_L2>
void computeNearestNeighborDistances(const std::vector<std::vector<double> >& dataset,
    const KDTreeVectorOfVectorsAdaptor<VectorOfVectorsType, num_t, DIM, Distance>& matIndex,
    std::vector<double>& NNDistances) {
    size_t k = 2;
    size_t dim = dataset[0].size();
    size_t total = dataset.size();
    std::vector<size_t> neighborIndices(k);
    std::vector<double> squaredDistances(k);

    for (size_t i = 0; i < total; ++i) {
        matIndex.query(&dataset[i][0], k, &neighborIndices[0], &squaredDistances[0]);

        size_t neighborIndex;
        double nnDistance;
        if (neighborIndices[0] == i) {
            neighborIndex = neighborIndices[k - 1];
            nnDistance = std::sqrt(squaredDistances[k - 1]);
        }
        else {
            neighborIndex = neighborIndices[0];
            nnDistance = std::sqrt(squaredDistances[0]);
        }

        if (nnDistance == 0.0) {
            size_t j = 0;
            size_t similarFeatureCount = std::accumulate(dataset[i].begin(), dataset[i].end(), 0,
                [&j, &dataset, neighborIndex](size_t currentSum, double featureValue) {
                    return currentSum + (featureValue == dataset[neighborIndex][j++]);
                }
            );
            
            if (similarFeatureCount == dim) {
                std::cout << "Duplicate datapoints in training set\n";
                std::exit(0);
            }
        }
        else {
            NNDistances.push_back(nnDistance);
        }
    }
}

// Compute weighted nearest neighbor distances for a class in the train dataset
void computeClassWeightedNearestNeighborDistances(const std::vector<std::vector<double> >& dataset,
    const std::string& className, std::unordered_map<std::string, std::vector<double> >& classNNDistMap) {
    size_t dim = dataset[0].size();

    if (dim <= 3) {
        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double, -1,
            metric_Weighted_L2_Simple> matIndex(dim, dataset, 10, 0, MODEL_STATE.featureWeights.at(className));
        computeNearestNeighborDistances(dataset, matIndex, classNNDistMap[className]);
    }
    else {
        KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double, -1,
            metric_Weighted_L2> matIndex(dim, dataset, 10, 0, MODEL_STATE.featureWeights.at(className));
        computeNearestNeighborDistances(dataset, matIndex, classNNDistMap[className]);
    }
}

// Compute weighted nearest neighbor distances for the train dataset
std::unordered_map<std::string, std::vector<double> > computeDatasetWeightedNearestNeighborDistances(
    const std::unordered_map<std::string, std::vector<std::vector<double> > >& classMap) {
    std::unordered_map<std::string, std::vector<double> > classNNDistMap;

    for (const auto& pair : classMap) {
        computeClassWeightedNearestNeighborDistances(pair.second, pair.first, classNNDistMap);
    }

    return classNNDistMap;
}

std::unordered_map<std::string, std::vector<double> > process(std::unordered_map<std::string,
    std::vector<ClassMember> >& dataset, int numNeighborsChecked, int minSameClassCount, size_t iteration) {
   
    /*
    // Beginning of saving original datapoints before standardization
    std::vector<std::string> header = { "className" };
    size_t dim = dataset.begin()->second[0].features.size();
    for (size_t i = 0; i < dim; ++i) {
        header.push_back("col" + std::to_string(i));
    }

    std::vector<std::vector<std::string> > rows;
    for (const auto& pair : dataset) {
        for (const ClassMember& obj : pair.second) {
            std::vector<std::string> row = { pair.first };
            std::for_each(obj.features.begin(), obj.features.end(), [&row](double val) {
                row.push_back(std::to_string(val));
                });
            rows.push_back(row);
        }
    }

    writeToCSV(header, rows, "iter" + std::to_string(iteration) + "-original.csv");
    // Ending of saving original datapoints before standardization
    */
    
    // Z-score standardization of features
   standardizeFeatures(dataset);

   /*
   // Beginning of saving original datapoints
   std::vector<std::string> header = { "className" };
   size_t dim = dataset.begin()->second[0].features.size();
   for (size_t i = 0; i < dim; ++i) {
       header.push_back("col" + std::to_string(i));
   }

   std::vector<std::vector<std::string> > rows;
   for (const auto& pair : dataset) {
       for (const ClassMember& obj : pair.second) {
           std::vector<std::string> row = { pair.first };
           std::for_each(obj.features.begin(), obj.features.end(), [&row](double val) {
               row.push_back(std::to_string(val));
               });
           rows.push_back(row);
       }
   }

   writeToCSV(header, rows, "iter" + std::to_string(iteration) + "-original.csv");
   // Ending of saving original datapoints
   */

   std::unordered_map<std::string, std::vector<std::vector<double> > > filteredDataset;
   if (MODEL_STATE.processType == "featureWeighting") {
       weightFeatures(dataset);
       filteredDataset = weightedFilterDatapoints(dataset, numNeighborsChecked, minSameClassCount);
   }
   else {
       filteredDataset = filterDatapoints(dataset, numNeighborsChecked, minSameClassCount);
   }

   /*
   std::unordered_map<std::string, std::vector<std::vector<double> > > filteredDataset = filterDatapoints(dataset,
        numNeighborsChecked, minSameClassCount);
    */

   /*
   std::cout << "Feature weights:\n";
   for (const auto& pair : MODEL_STATE.featureWeights) {
       std::cout << pair.first << ": ";
       std::copy(pair.second.begin(), pair.second.end(), std::ostream_iterator<double>(std::cout, ", "));
       std::cout << std::endl;
   }
   */

   /*
   // Beginning of saving filtered datapoints
   rows.clear();
   for (const auto& pair : filteredDataset) {
       for (const std::vector<double>& datapoint : pair.second) {
           std::vector<std::string> row = { pair.first };
           std::for_each(datapoint.begin(), datapoint.end(), [&row](double val) {
               row.push_back(std::to_string(val));
               });
           rows.push_back(row);
       }
   }

   writeToCSV(header, rows, "iter" + std::to_string(iteration) + "-filtered.csv");
   // Ending of saving filtered datapoints
   */
   
   // compute k nearest distance, k = 1
   std::unordered_map<std::string, std::vector<double> > classNNDistMap;
   if (MODEL_STATE.processType == "featureWeighting") {
       classNNDistMap = computeDatasetWeightedNearestNeighborDistances(filteredDataset);
   }
   else {
       classNNDistMap = computeNearestNeighborDistances(filteredDataset);
   }
   
   // Save all datapoints for each class
   MODEL_STATE.classMap = std::move(filteredDataset);

   std::vector<std::string> warningClasses;

   /*
   // sort distances in ascending order
   for (auto& pair : classNNDistMap) {
       std::sort(pair.second.begin(), pair.second.end());

       // eliminate duplicated results
       pair.second.erase(unique(pair.second.begin(), pair.second.end()), pair.second.end());

       if (pair.second.size() < MIN_CLASS_MEMBERS) {
           std::cout << "Warning: Class \"" << pair.first << "\" has an insufficient amount "
               "of unique points for its ECDF\n";
           MODEL_STATE.classMap.erase(pair.first);
           continue;
       }

       if (pair.second[0] > 1.0) {
           warningClasses.push_back(pair.first);
       }
   }
   */

   // sort distances in ascending order
   for (auto iter = classNNDistMap.begin(); iter != classNNDistMap.end();) {
       std::sort(iter->second.begin(), iter->second.end());

       // eliminate duplicated results
       iter->second.erase(unique(iter->second.begin(), iter->second.end()), iter->second.end());

       if (iter->second.size() < MIN_CLASS_MEMBERS) {
           std::cout << "Warning: Class \"" << iter->first << "\" has an insufficient amount "
               "of unique points for its ECDF\n";
           MODEL_STATE.classMap.erase(iter->first);
           iter = classNNDistMap.erase(iter);
           continue;
       }

       if (iter->second[0] == 0.0) {
           std::cout << "Duplicate datapoints in training set\n";
           std::exit(0);
       }

       if (iter->second[0] > 1.0) {
           warningClasses.push_back(iter->first);
       }

       ++iter;
   }

   if (warningClasses.size() > 0) {
       std::cout << "Nearest neighbor distances are greater than 1 for these classes: ";
       std::copy(warningClasses.begin(), warningClasses.end(), std::ostream_iterator<std::string>(std::cout, ", "));
       std::cout << std::endl;
   }

   if (classNNDistMap.size() <= 1) {
       std::cout << classNNDistMap.size() << " classes have an ECDF with at least " <<
           MIN_CLASS_MEMBERS << " points: ";
       if (classNNDistMap.size() == 1) {
           std::cout << classNNDistMap.begin()->first << std::endl;
       }
       std::exit(0);
   }

   return classNNDistMap;
}
