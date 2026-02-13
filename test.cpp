#include <cstdlib>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <iterator>
#include <thread>
#include <cassert>
#include <numeric>
#include "ap.h"

#include "classMember.h"
#include "process.h"
#include "modelState.h"
#include "fit.h"
#include "test.h"
#include "CSVWrite.hpp"
#include "testResults.h"
#include "cachePaths.h"
#include "NOTAPoints.h"
#include "saveResults.h"

void distributeAcrossFolds(const std::unordered_map<std::string, std::vector<ClassMember> >::const_iterator& datasetIter, std::unordered_map<std::string, std::vector<ClassMember> > kSets[]) {
	auto& datapoints = datasetIter->second;
	size_t minDatapointsPerFold = datapoints.size() / K_FOLDS;

	size_t extraDatapoints = datapoints.size() - minDatapointsPerFold * K_FOLDS;
	std::vector<ClassMember>::const_iterator datapointsIter = datapoints.begin();

	for (int i = 0; i < K_FOLDS; ++i) {
		size_t datapointsToInsert = (i < extraDatapoints) ? minDatapointsPerFold + 1 : minDatapointsPerFold;
		std::vector<ClassMember>& foldClassData = kSets[i][datasetIter->first];
		foldClassData.reserve(datapointsToInsert);
		foldClassData.insert(foldClassData.end(), datapointsIter, datapointsIter + datapointsToInsert);
		datapointsIter += datapointsToInsert;
	}
}

void kFoldSplit(std::unordered_map<std::string, std::vector<ClassMember> >& dataset,
	std::unordered_map<std::string, std::vector<ClassMember> > kSets[], size_t maxPerClass) {
	std::random_device rand_d;

	std::mt19937 gen(rand_d());
	size_t minPerClass = std::ceil(static_cast<double>(MIN_CLASS_MEMBERS) / static_cast<double>((K_FOLDS - 1))) * K_FOLDS;
	for (auto iter = dataset.begin(); iter != dataset.end();) {
		auto& members = iter->second;
		if (members.size() < minPerClass) {
			std::cout << "Warning: Removing \"" << iter->first << "\" from classes, because "
				"there is an insufficient number of instances to have at least " <<
				MIN_CLASS_MEMBERS << " points in the ECDF across " << K_FOLDS << " folds\n";
			iter = dataset.erase(iter);
			continue;
		}
		std::shuffle(members.begin(), members.end(), gen);
		if (members.size() > maxPerClass) {
			members.resize(maxPerClass);
		}
		distributeAcrossFolds(iter, kSets);
		MODEL_STATE.classNames.push_back(iter->first);
		MODEL_STATE.numInstancesPerClass[iter->first] = members.size();
		++iter;
	}

	// Check if there are 0 or 1 classes to train with
	if (MODEL_STATE.classNames.size() <= 1) {
		std::cout << MODEL_STATE.classNames.size() << " classes have enough instances: ";
		if (MODEL_STATE.classNames.size() == 1) {
			std::cout << MODEL_STATE.classNames[0] << std::endl;
		}
		std::exit(0);
	}

	std::sort(MODEL_STATE.classNames.begin(), MODEL_STATE.classNames.end());
}

std::unordered_map<std::string, double[PVALUE_NUMERATOR_MAX] > createPerClassPValueThresholds() {
	std::unordered_map<std::string, double[PVALUE_NUMERATOR_MAX]> pvalueThresholds;

	for (const auto& pair : MODEL_STATE.numInstancesPerClass) {
		for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
			pvalueThresholds[pair.first][pvalCat] = (pvalCat + 1) / pair.second;
		}
	}

	return pvalueThresholds;
}

/*
std::unordered_map<std::string, double> createPerClassPValueThresholds() {
	std::unordered_map<std::string, double> pvalueThresholds;

	for (const auto& pair : MODEL_STATE.numInstancesPerClass) {
		pvalueThresholds[pair.first] = 1 / pair.second;
	}

	return pvalueThresholds;
}
*/

void setPValueThreshold(const std::string& threshold) {
	if (threshold == "default") {
		TEST_RESULTS.pvalueThreshold = 1.0 / MODEL_STATE.datasetSize;
	}
	else if (threshold == "geometric mean") {		
		double total = std::accumulate(MODEL_STATE.numInstancesPerClass.begin(), MODEL_STATE.numInstancesPerClass.end(),
			1.0, [](double a, const std::pair<std::string, double>& b) { return a * (1.0 / b.second); });
		TEST_RESULTS.pvalueThreshold = std::pow(total, 1.0 / MODEL_STATE.numInstancesPerClass.size());
	}
	else if (threshold == "per class") {
		TEST_RESULTS.perClassThresholds = createPerClassPValueThresholds();
	}
	else {
		TEST_RESULTS.pvalueThreshold = std::stod(threshold);
	}
}

std::vector<ClassMember> standardize(std::vector<ClassMember> dataset) {
	if (!MODEL_STATE.zeroStdDeviation.empty()) {
		removeFeatures(MODEL_STATE.zeroStdDeviation, dataset);
	}
	size_t numFeatures = dataset[0].features.size();
	for (auto& obj : dataset) {
		if (obj.features.size() != numFeatures) {
			std::cerr << "Inconsistent feature size: " << numFeatures << " != " << obj.features.size() << std::endl;
			std::exit(0);
		}
		for (size_t i = 0; i < numFeatures; ++i) {
			obj.features[i] = (obj.features[i] - MODEL_STATE.means[i]) / MODEL_STATE.sigmas[i];
		}
	}

	return dataset;
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

// Project test dataset into lower dimension subspace
void toPCASubspace(std::vector<ClassMember>& dataset) {
	alglib::real_2d_array datapoints, principalComponents;
	datapoints.setlength(dataset.size(), dataset[0].features.size());
	principalComponents.setlength(dataset.size(), MODEL_STATE.principalAxes.cols());
	copyDatapoints(dataset, datapoints, true);
	projectOntoPrincipalAxes(datapoints, MODEL_STATE.principalAxes, principalComponents);
	copyDatapoints(dataset, principalComponents, false);
}

// Calculate the minimum distance between the datapoints in the test set with each class
void computeClassDistances(const std::vector<ClassMember>& dataset, std::vector<std::unordered_map<std::string, double> >& nnDistances,
	size_t start, size_t stop) {
	for (size_t i = start; i < stop; ++i) {
		std::unordered_map<std::string, double> classDistance;
		for (const auto& pair : MODEL_STATE.classMap) {
			double minDistance = std::numeric_limits<double>::max();
			for (const auto& classDatapoint : pair.second) {
				double distance = (MODEL_STATE.processType == "featureWeighting") ?
					weightedEuclideanDistance(dataset[i].features, classDatapoint, MODEL_STATE.featureWeights.at(pair.first)) :
					euclideanDistance(dataset[i].features, classDatapoint);
				if (distance < minDistance) {
					minDistance = distance;
				}
			}
			classDistance[pair.first] = minDistance;
		}
		nnDistances[i] = std::move(classDistance);
	}
}

// Calculate the p values of the datapoints in the test set for each class between start and stop
void calculatePValuesInRange(const std::vector<std::unordered_map<std::string, double> >& dataset,
	std::vector<std::unordered_map<std::string, double> >& pvalues, size_t start, size_t stop) {
	for (size_t i = start; i < stop; ++i) {
		std::unordered_map<std::string, double> result;
		for (const auto& pair : dataset[i]) {
			if (MODEL_STATE.bestFit.find(pair.first) == MODEL_STATE.bestFit.end()) {
				continue;
			}

			std::string bestFitFunction = MODEL_STATE.bestFit.at(pair.first).functionName;
			double c = MODEL_STATE.bestFit[pair.first].c[0], a = MODEL_STATE.bestFit[pair.first].c[1];
			double pvalue;
			if (bestFitFunction == "Logistic function") {
				pvalue = 1 - logistic(c, a, pair.second);
			}
			else if (bestFitFunction == "hyperbolic tangent function") {
				pvalue = 1 - hyperbolic_tangent(c, a, pair.second);
			}
			else if (bestFitFunction == "arctangent function") {
				pvalue = 1 - arctangent(c, a, pair.second);
			}
			else if (bestFitFunction == "gudermannian function") {
				pvalue = 1 - gudermannian(c, a, pair.second);
			}
			else if (bestFitFunction == "Gompertz function") {
				pvalue = 1 - gompertz(c, a, pair.second);
			}
			else if (bestFitFunction == "error function based sigmoid") {
				pvalue = 1 - erf_sigmoid(c, a, pair.second);
			}
			else {
				pvalue = 1 - algebraic(c, a, pair.second);
			}
			result[pair.first] = pvalue;
		}

		m.lock();
		pvalues[i] = std::move(result);
		m.unlock();
	}
}

std::vector<std::unordered_map<std::string, double> > calculatePValues(
	const std::vector<std::unordered_map<std::string, double> >& nnDistances) {
	size_t start = 0, stop = 0;
	size_t total = nnDistances.size();
	size_t datapointsPerThread = std::round(total / NUM_THREADS);
	std::vector<std::thread> threads;
	std::vector<std::unordered_map<std::string, double> > pvalues(total);	

	for (size_t i = 0; i < NUM_THREADS && stop != total; ++i) {
		if (i == NUM_THREADS - 1) {
			stop = total;
		}
		else {
			stop = std::min(total, stop + datapointsPerThread);
		}

		threads.emplace_back(std::thread{ calculatePValuesInRange, std::cref(nnDistances), std::ref(pvalues), start, stop });

		start = stop;
	}

	for (auto& t : threads) {
		t.join();
	}

	return pvalues;
}

//Print out the p values for every datapoint in the test set
void printPValues(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues) {
	std::cout << "P values for test set:\n";
	for (size_t i = 0; i < dataset.size(); ++i) {
		// Print out the feature values for the datapoint
		std::cout << "Feature values: ";
		std::copy(dataset[i].features.begin(), dataset[i].features.end(), std::ostream_iterator<double>(std::cout, ", "));
		std::cout << std::endl;

		// Print out the p values of the datapoint for each class
		for (const auto& pair : pvalues[i]) {
			std::cout << pair.first << ": " << pair.second << std::endl;
		}

		// Print out the actual class of the datapoint
		//std::string expectedClass = (dataset[i].name == "NOTA") ? "Injected \"none of the above\" point" : dataset[i].name;
		std::string expectedClass = dataset[i].name;
		if (dataset[i].name == "NOTA") {
			if (dataset[i].NOTALocation == NOTACategory::VOID) {
				expectedClass = "Void" + std::to_string(dataset[i].NNUnitsFromClass);
			}
			else {
				expectedClass = "Hyperspace" + std::to_string(dataset[i].NOTALocation);
			}
		}
		std::cout << "Actual class: " << expectedClass << std::endl;
	}
}

/*
// Increment the overall prediction statistics for all the different NOTA point categories
void incrementOverallPredStats(size_t idx) {
	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		++TEST_RESULTS.overallHyperspacePredStats[static_cast<NOTACategory>(i)][idx];
	}
	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		++TEST_RESULTS.overallVoidPredStats[i][idx];
	}
}

// Increment the confusion matrix of a class for all the different NOTA point categories
void incrementAllConfusionMatrices(const std::string& className, size_t idx) {
	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		++TEST_RESULTS.hyperspaceConfusionMatrix[static_cast<NOTACategory>(i)][className][idx];
	}
	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		++TEST_RESULTS.voidConfusionMatrix[i][className][idx];
	}
}
*/

// Increment the overall prediction statistics for all the different NOTA point categories
void incrementOverallPredStats(size_t pvalCat, size_t idx) {
	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		++TEST_RESULTS.overallHyperspacePredStats[pvalCat][static_cast<NOTACategory>(i)][idx];
	}
	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		++TEST_RESULTS.overallVoidPredStats[pvalCat][i][idx];
	}
}

// Increment the confusion matrix of a class for all the different NOTA point categories
void incrementAllConfusionMatrices(const std::string& className, size_t pvalCat, size_t idx) {
	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		++TEST_RESULTS.hyperspaceConfusionMatrix[pvalCat][static_cast<NOTACategory>(i)][className][idx];
	}
	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		++TEST_RESULTS.voidConfusionMatrix[pvalCat][i][className][idx];
	}
}

// Previous version that work using just one p value threshold
/*
std::vector<std::pair<std::string, std::string> > calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues, double pvalueThreshold) {
	size_t total = dataset.size();
	std::vector<std::pair<std::string, std::string> > classifications(total);

	for (size_t i = 0; i < total; ++i) {
		std::string expectedClass = dataset[i].name;

		// If the current point is a NOTA point, adjust the name to reflect the specific category
		if (expectedClass == "NOTA") {
			if (dataset[i].NOTALocation == NOTACategory::VOID) {
				expectedClass = "Void" + std::to_string(static_cast<int>(dataset[i].NNUnitsFromClass));
			}
			else {
				expectedClass = "Hyperspace" + std::to_string(dataset[i].NOTALocation);
			}
		}

		// Find the largest p value for the datapoint
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA) for overall statistics
		// Tally whether p value indicates a true positive, false positive, or false negative for a class
		if (largestPValue->second >= pvalueThreshold) {
			if (dataset[i].name == "NOTA") {
				if (dataset[i].NOTALocation == NOTACategory::VOID) {
					++TEST_RESULTS.voidConfusionMatrix[dataset[i].NNUnitsFromClass][largestPValue->first][2];
					++TEST_RESULTS.voidRandomPoints[dataset[i].NNUnitsFromClass][2];
				}
				else {
					++TEST_RESULTS.hyperspaceConfusionMatrix[dataset[i].NOTALocation][largestPValue->first][2];
					++TEST_RESULTS.hyperspaceRandomPoints[dataset[i].NOTALocation][2];
				}
			}
			else if (largestPValue->first == expectedClass) {
				incrementOverallPredStats(0);
				incrementAllConfusionMatrices(largestPValue->first, 0);
			}
			else {
				incrementOverallPredStats(1);
				incrementAllConfusionMatrices(largestPValue->first, 2);
				incrementAllConfusionMatrices(expectedClass, 1);
			}
			classifications[i] = { expectedClass, largestPValue->first };
		}
		else {
			if (dataset[i].name == "NOTA") {
				if (dataset[i].NOTALocation == NOTACategory::VOID) {
					++TEST_RESULTS.voidRandomPoints[dataset[i].NNUnitsFromClass][1];
				}
				else {
					++TEST_RESULTS.hyperspaceRandomPoints[dataset[i].NOTALocation][1];
				}
			}
			else {
				incrementOverallPredStats(2);
				incrementAllConfusionMatrices(expectedClass, 1);
				for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
					++TEST_RESULTS.hyperspaceRandomPoints[static_cast<NOTACategory>(i)][3];
				}
				for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
					++TEST_RESULTS.voidRandomPoints[i][3];
				}
			}
			classifications[i] = { expectedClass, "NOTA" };
		}
	}

	return classifications;
}
*/

// Previous version that work with one type of per class p value threshold
/*
std::vector<std::pair<std::string, std::string> > calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues) {
	size_t total = dataset.size();
	std::vector<std::pair<std::string, std::string> > classifications(total);
	double pvalueThreshold;

	for (size_t i = 0; i < total; ++i) {
		std::string expectedClass = dataset[i].name;

		// If the current point is a NOTA point, adjust the name to reflect the specific category
		if (expectedClass == "NOTA") {
			if (dataset[i].NOTALocation == NOTACategory::VOID) {
				expectedClass = "Void" + std::to_string(static_cast<int>(dataset[i].NNUnitsFromClass));
			}
			else {
				expectedClass = "Hyperspace" + std::to_string(dataset[i].NOTALocation);
			}
		}

		// Find the largest p value for the datapoint
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		pvalueThreshold = (TEST_RESULTS.pvalueThreshold >= 0) ? TEST_RESULTS.pvalueThreshold : TEST_RESULTS.perClassThresholds.at(largestPValue->first);

		// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA) for overall statistics
		// Tally whether p value indicates a true positive, false positive, or false negative for a class
		if (largestPValue->second >= pvalueThreshold) {
			if (dataset[i].name == "NOTA") {
				if (dataset[i].NOTALocation == NOTACategory::VOID) {
					++TEST_RESULTS.voidConfusionMatrix[dataset[i].NNUnitsFromClass][largestPValue->first][2];
					++TEST_RESULTS.voidRandomPoints[dataset[i].NNUnitsFromClass][2];
				}
				else {
					++TEST_RESULTS.hyperspaceConfusionMatrix[dataset[i].NOTALocation][largestPValue->first][2];
					++TEST_RESULTS.hyperspaceRandomPoints[dataset[i].NOTALocation][2];
				}
			}
			else if (largestPValue->first == expectedClass) {
				incrementOverallPredStats(0);
				incrementAllConfusionMatrices(largestPValue->first, 0);
			}
			else {
				incrementOverallPredStats(1);
				incrementAllConfusionMatrices(largestPValue->first, 2);
				incrementAllConfusionMatrices(expectedClass, 1);
			}
			classifications[i] = { expectedClass, largestPValue->first };
		}
		else {
			if (dataset[i].name == "NOTA") {
				if (dataset[i].NOTALocation == NOTACategory::VOID) {
					++TEST_RESULTS.voidRandomPoints[dataset[i].NNUnitsFromClass][1];
				}
				else {
					++TEST_RESULTS.hyperspaceRandomPoints[dataset[i].NOTALocation][1];
				}
			}
			else {
				incrementOverallPredStats(2);
				incrementAllConfusionMatrices(expectedClass, 1);
				for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
					++TEST_RESULTS.hyperspaceRandomPoints[static_cast<NOTACategory>(i)][3];
				}
				for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
					++TEST_RESULTS.voidRandomPoints[i][3];
				}
			}
			classifications[i] = { expectedClass, "NOTA" };
		}
	}

	
	std::cout << "Hyperspace 40:\n";
	printf("Overall prediction statistics: %f, %f, %f\n", TEST_RESULTS.overallHyperspacePredStats[NOTACategory::HYPERSPACE40][0],
		TEST_RESULTS.overallHyperspacePredStats[NOTACategory::HYPERSPACE40][1], TEST_RESULTS.overallHyperspacePredStats[NOTACategory::HYPERSPACE40][2]);
	printf("Confusion Matrices: \n");
	for (const auto& pair : TEST_RESULTS.hyperspaceConfusionMatrix[NOTACategory::HYPERSPACE40]) {
		std::cout << pair.first << ": ";
		for (size_t i = 0; i < 3; ++i)
			std::cout << pair.second[i] << ", ";
		std::cout << std::endl;
	}
	std::cout << "NOTA Points: ";
	for (size_t i = 1; i < 5; ++i) std::cout << TEST_RESULTS.hyperspaceRandomPoints[NOTACategory::HYPERSPACE40][i] << ", ";
	std::cout << std::endl;

	std::cout << "Void 0.6:\n";
	std::cout << "Overall prediction statistics: ";
	for (size_t i = 0; i < 3; ++i) std::cout << TEST_RESULTS.overallVoidPredStats[5][i] << ", ";
	std::cout << std::endl;
	std::cout << "Confusion Matrices: \n";
	for (const auto& pair : TEST_RESULTS.voidConfusionMatrix[5]) {
		std::cout << pair.first << ": ";
		for (size_t i = 0; i < 3; ++i)
			std::cout << pair.second[i] << ", ";
		std::cout << std::endl;
	}
	std::cout << "NOTA Points: ";
	for (size_t i = 1; i < 5; ++i) std::cout << TEST_RESULTS.voidRandomPoints[5][i] << ", ";
	std::cout << std::endl;
	

	return classifications;
}
*/

/*
std::vector<std::pair<std::string, std::string> > calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues) {
	std::vector<std::pair<std::string, std::string> > classifications;

	//classifications = calculateStatistics(dataset, pvalues, TEST_RESULTS.pvalueThreshold);
	
	if (TEST_RESULTS.pvalueThreshold >= 0) {
		classifications = calculateStatistics(dataset, pvalues, TEST_RESULTS.pvalueThreshold);
	}
	else {
		classifications = calculateStatistics(dataset, pvalues, TEST_RESULTS.perClassThresholds);
	}
	

	return classifications;
}
*/

std::vector<std::pair<std::string, std::string> > calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues) {
	size_t total = dataset.size();
	std::vector<std::pair<std::string, std::string> > classifications(total);
	double pvalueThreshold;

	for (size_t i = 0; i < total; ++i) {
		std::string expectedClass = dataset[i].name;

		// If the current point is a NOTA point, adjust the name to reflect the specific category
		if (expectedClass == "NOTA") {
			if (dataset[i].NOTALocation == NOTACategory::VOID) {
				expectedClass = "Void" + std::to_string(static_cast<int>(dataset[i].NNUnitsFromClass));
			}
			else {
				expectedClass = "Hyperspace" + std::to_string(dataset[i].NOTALocation);
			}
		}

		// Find the largest p value for the datapoint
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
			pvalueThreshold = (TEST_RESULTS.pvalueThreshold >= 0) ? TEST_RESULTS.pvalueThreshold : TEST_RESULTS.perClassThresholds.at(largestPValue->first)[pvalCat];

			// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA) for overall statistics
			// Tally whether p value indicates a true positive, false positive, or false negative for a class
			if (largestPValue->second >= pvalueThreshold) {
				if (dataset[i].name == "NOTA") {
					if (dataset[i].NOTALocation == NOTACategory::VOID) {
						++TEST_RESULTS.voidConfusionMatrix[pvalCat][dataset[i].NNUnitsFromClass][largestPValue->first][2];
						++TEST_RESULTS.voidRandomPoints[pvalCat][dataset[i].NNUnitsFromClass][2];
					}
					else {
						++TEST_RESULTS.hyperspaceConfusionMatrix[pvalCat][dataset[i].NOTALocation][largestPValue->first][2];
						++TEST_RESULTS.hyperspaceRandomPoints[pvalCat][dataset[i].NOTALocation][2];
					}
				}
				else if (largestPValue->first == expectedClass) {
					incrementOverallPredStats(pvalCat, 0);
					incrementAllConfusionMatrices(largestPValue->first, pvalCat, 0);
				}
				else {
					incrementOverallPredStats(pvalCat, 1);
					incrementAllConfusionMatrices(largestPValue->first, pvalCat, 2);
					incrementAllConfusionMatrices(expectedClass, pvalCat, 1);
				}
			}
			else {
				if (dataset[i].name == "NOTA") {
					if (dataset[i].NOTALocation == NOTACategory::VOID) {
						++TEST_RESULTS.voidRandomPoints[pvalCat][dataset[i].NNUnitsFromClass][1];
					}
					else {
						++TEST_RESULTS.hyperspaceRandomPoints[pvalCat][dataset[i].NOTALocation][1];
					}
				}
				else {
					incrementOverallPredStats(pvalCat, 2);
					incrementAllConfusionMatrices(expectedClass, pvalCat, 1);
					for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
						++TEST_RESULTS.hyperspaceRandomPoints[pvalCat][static_cast<NOTACategory>(i)][3];
					}
					for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
						++TEST_RESULTS.voidRandomPoints[pvalCat][i][3];
					}
				}
			}
		}
	}
		
	/*
	size_t testPvalCat = 4;
	std::cout << "Hyperspace 40:\n";
	printf("Overall prediction statistics: %f, %f, %f\n", TEST_RESULTS.overallHyperspacePredStats[testPvalCat][NOTACategory::HYPERSPACE40][0],
		TEST_RESULTS.overallHyperspacePredStats[testPvalCat][NOTACategory::HYPERSPACE40][1], TEST_RESULTS.overallHyperspacePredStats[testPvalCat][NOTACategory::HYPERSPACE40][2]);
	printf("Confusion Matrices: \n");
	for (const auto& pair : TEST_RESULTS.hyperspaceConfusionMatrix[testPvalCat][NOTACategory::HYPERSPACE40]) {
		std::cout << pair.first << ": ";
		for (size_t i = 0; i < 3; ++i)
			std::cout << pair.second[i] << ", ";
		std::cout << std::endl;
	}
	std::cout << "NOTA Points: ";
	for (size_t i = 1; i < 5; ++i) std::cout << TEST_RESULTS.hyperspaceRandomPoints[testPvalCat][NOTACategory::HYPERSPACE40][i] << ", ";
	std::cout << std::endl;

	testPvalCat = 7;
	std::cout << "Void 0.6:\n";
	std::cout << "Overall prediction statistics: ";
	for (size_t i = 0; i < 3; ++i) std::cout << TEST_RESULTS.overallVoidPredStats[testPvalCat][5][i] << ", ";
	std::cout << std::endl;
	std::cout << "Confusion Matrices: \n";
	for (const auto& pair : TEST_RESULTS.voidConfusionMatrix[testPvalCat][5]) {
		std::cout << pair.first << ": ";
		for (size_t i = 0; i < 3; ++i)
			std::cout << pair.second[i] << ", ";
		std::cout << std::endl;
	}
	std::cout << "NOTA Points: ";
	for (size_t i = 1; i < 5; ++i) std::cout << TEST_RESULTS.voidRandomPoints[testPvalCat][5][i] << ", ";
	std::cout << std::endl;
	*/

	return classifications;
}

/*
void calculateVoidNOTAPointsSummary(size_t idx) {
	// Determine size of entire dataset used for training/testing plus "randomly placed" points
	size_t total = MODEL_STATE.datasetSize + TEST_RESULTS.voidRandomPoints[idx][0];

	for (const std::string& className : MODEL_STATE.classNames) {
		// Calculate number of true negatives for each class
		double* classConfusionMatrix = TEST_RESULTS.voidConfusionMatrix[idx][className];
		classConfusionMatrix[3] = total - classConfusionMatrix[0] - classConfusionMatrix[1] - classConfusionMatrix[2];

		// Calculate the precision for each class
		TEST_RESULTS.voidPrecision[idx][className] = (classConfusionMatrix[0] + classConfusionMatrix[2] == 0) ?
			0 : (classConfusionMatrix[0] / (classConfusionMatrix[0] + classConfusionMatrix[2])) * 100;
	}

	for (size_t i = 0; i < 3; ++i) {
		TEST_RESULTS.overallVoidPredStats[idx][i] /= MODEL_STATE.datasetSize;
		TEST_RESULTS.overallVoidPredStats[idx][i] *= 100;
	}

	// Calculate number of true negatives and recall for NOTA points
	double* randomPoints = TEST_RESULTS.voidRandomPoints[idx];
	randomPoints[4] = total - randomPoints[1] - randomPoints[2] - randomPoints[3];
	randomPoints[5] = (randomPoints[1] / (randomPoints[1] + randomPoints[2])) * 100;
}
*/

void calculateVoidNOTAPointsSummary(size_t pvalCat, size_t idx) {
	// Determine size of entire dataset used for training/testing plus "randomly placed" points
	size_t total = MODEL_STATE.datasetSize + TEST_RESULTS.voidRandomPoints[pvalCat][idx][0];

	for (const std::string& className : MODEL_STATE.classNames) {
		// Calculate number of true negatives for each class
		double* classConfusionMatrix = TEST_RESULTS.voidConfusionMatrix[pvalCat][idx][className];
		classConfusionMatrix[3] = total - classConfusionMatrix[0] - classConfusionMatrix[1] - classConfusionMatrix[2];

		// Calculate the precision for each class
		TEST_RESULTS.voidPrecision[pvalCat][idx][className] = (classConfusionMatrix[0] + classConfusionMatrix[2] == 0) ?
			0 : (classConfusionMatrix[0] / (classConfusionMatrix[0] + classConfusionMatrix[2])) * 100;
	}

	for (size_t i = 0; i < 3; ++i) {
		TEST_RESULTS.overallVoidPredStats[pvalCat][idx][i] /= MODEL_STATE.datasetSize;
		TEST_RESULTS.overallVoidPredStats[pvalCat][idx][i] *= 100;
	}

	// Calculate number of true negatives and recall for NOTA points
	double* randomPoints = TEST_RESULTS.voidRandomPoints[pvalCat][idx];
	randomPoints[4] = total - randomPoints[1] - randomPoints[2] - randomPoints[3];
	randomPoints[5] = (randomPoints[1] / (randomPoints[1] + randomPoints[2])) * 100;
}

/*
void calculateHyperspaceNOTAPointsSummary(NOTACategory NOTALoc) {
	// Determine size of entire dataset used for training/testing plus "randomly placed" points
	size_t total = MODEL_STATE.datasetSize + TEST_RESULTS.hyperspaceRandomPoints[NOTALoc][0];

	for (const std::string& className : MODEL_STATE.classNames) {
		// Calculate number of true negatives for each class
		double* classConfusionMatrix = TEST_RESULTS.hyperspaceConfusionMatrix[NOTALoc][className];
		classConfusionMatrix[3] = total - classConfusionMatrix[0] - classConfusionMatrix[1] - classConfusionMatrix[2];

		// Calculate the precision for each class
		TEST_RESULTS.hyperspacePrecision[NOTALoc][className] = (classConfusionMatrix[0] + classConfusionMatrix[2] == 0) ?
			0 : (classConfusionMatrix[0] / (classConfusionMatrix[0] + classConfusionMatrix[2])) * 100;
	}

	for (size_t i = 0; i < 3; ++i) {
		TEST_RESULTS.overallHyperspacePredStats[NOTALoc][i] /= MODEL_STATE.datasetSize;
		TEST_RESULTS.overallHyperspacePredStats[NOTALoc][i] *= 100;
	}

	// Calculate number of true negatives and recall for NOTA points
	double* randomPoints = TEST_RESULTS.hyperspaceRandomPoints[NOTALoc];
	randomPoints[4] = total - randomPoints[1] - randomPoints[2] - randomPoints[3];
	randomPoints[5] = (randomPoints[1] / (randomPoints[1] + randomPoints[2])) * 100;
}
*/

void calculateHyperspaceNOTAPointsSummary(size_t pvalCat, NOTACategory NOTALoc) {
	// Determine size of entire dataset used for training/testing plus "randomly placed" points
	size_t total = MODEL_STATE.datasetSize + TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTALoc][0];

	for (const std::string& className : MODEL_STATE.classNames) {
		// Calculate number of true negatives for each class
		double* classConfusionMatrix = TEST_RESULTS.hyperspaceConfusionMatrix[pvalCat][NOTALoc][className];
		classConfusionMatrix[3] = total - classConfusionMatrix[0] - classConfusionMatrix[1] - classConfusionMatrix[2];

		// Calculate the precision for each class
		TEST_RESULTS.hyperspacePrecision[pvalCat][NOTALoc][className] = (classConfusionMatrix[0] + classConfusionMatrix[2] == 0) ?
			0 : (classConfusionMatrix[0] / (classConfusionMatrix[0] + classConfusionMatrix[2])) * 100;
	}

	for (size_t i = 0; i < 3; ++i) {
		TEST_RESULTS.overallHyperspacePredStats[pvalCat][NOTALoc][i] /= MODEL_STATE.datasetSize;
		TEST_RESULTS.overallHyperspacePredStats[pvalCat][NOTALoc][i] *= 100;
	}

	// Calculate number of true negatives and recall for NOTA points
	double* randomPoints = TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTALoc];
	randomPoints[4] = total - randomPoints[1] - randomPoints[2] - randomPoints[3];
	randomPoints[5] = (randomPoints[1] / (randomPoints[1] + randomPoints[2])) * 100;
}

/*
void calculateSummary() {
	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		calculateVoidNOTAPointsSummary(i);
	}
	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		calculateHyperspaceNOTAPointsSummary(static_cast<NOTACategory>(i));
	}
}
*/

void calculateSummary() {
	for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
		for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
			calculateVoidNOTAPointsSummary(pvalCat, i);
		}
		for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
			calculateHyperspaceNOTAPointsSummary(pvalCat, static_cast<NOTACategory>(i));
		}
	}
}

/*
void printPValueThreshold() {
	//printf("P value threshold: %g\n", TEST_RESULTS.pvalueThreshold);
	
	if (TEST_RESULTS.pvalueThreshold >= 0) {
		printf("P value threshold: %g\n", TEST_RESULTS.pvalueThreshold);
	}
	else {
		std::cout << "Per class p value thresholds:\n";
		for (const std::string& className : MODEL_STATE.classNames) {
			printf("%s: %g\n", className.c_str(), TEST_RESULTS.perClassThresholds.at(className));
		}
		std::cout << std::endl;
	}
	
}
*/

void printPValueThreshold(size_t pvalCat) {
	//printf("P value threshold: %g\n", TEST_RESULTS.pvalueThreshold);

	if (TEST_RESULTS.pvalueThreshold >= 0) {
		printf("P value threshold: %g\n", TEST_RESULTS.pvalueThreshold);
	}
	else {
		std::cout << "Per class p value thresholds:\n";
		for (const std::string& className : MODEL_STATE.classNames) {
			printf("%s: %g\n", className.c_str(), TEST_RESULTS.perClassThresholds.at(className)[pvalCat]);
		}
		std::cout << std::endl;
	}

}

void printPredCategories(double* overallPredStats) {
	int defaultPrecision = std::cout.precision();
	std::cout << std::setprecision(2) << std::fixed;
	std::cout << "Correct: " << overallPredStats[0] << "%\n";
	std::cout << "Incorrect: " << overallPredStats[1] << "%\n";
	std::cout << "None of the above: " << overallPredStats[2] << "%\n\n";
	std::cout << std::setprecision(defaultPrecision);
	std::cout.unsetf(std::ios::fixed);
}

void printConfusionMatrix(double confusionMatrix[]) {
	std::cout << "Confusion Matrix:\n";
	std::cout << "|---------|---------|" << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "| " << std::setw(8) << confusionMatrix[0] << "| " << std::setw(8) <<
		confusionMatrix[1] << "|" << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "|---------|---------|" << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "| " << std::setw(8) << confusionMatrix[2] << "| " << std::setw(8) <<
		confusionMatrix[3] << "|" << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(10) << "|" << std::setw(10) << "|" << std::left << std::endl;
	std::cout << "|---------|---------|" << std::endl << std::endl;
}

void printPredCategoriesPerClass(double* confusionMatrix, double precision) {
	printConfusionMatrix(confusionMatrix);

	// Print out precision for class
	int defaultPrecision = std::cout.precision();
	std::cout << "Precision: " << std::setprecision(2) << std::fixed << precision << "%\n\n";
	std::cout << std::setprecision(defaultPrecision);
	std::cout.unsetf(std::ios::fixed);
}

void printPredCategoriesAllClasses(std::unordered_map<std::string, double[4]>& confusionMatrices,
	const std::unordered_map<std::string, double>& precision) {
	for (const std::string& className : MODEL_STATE.classNames) {
		std::cout << "Class: " << className << std::endl;
		std::cout << "Total instances: " << MODEL_STATE.numInstancesPerClass.at(className) << std::endl;
		printPredCategoriesPerClass(confusionMatrices[className], precision.at(className));
	}
}

void printNOTAStatistics(double* NOTAPoints) {
	std::cout << "\"None of the above\" classifications:\n";
	std::cout << "Total randomly placed points: " << NOTAPoints[0] << std::endl;
	printConfusionMatrix(&NOTAPoints[1]);
	printf("Recall: %.2f%%\n\n", NOTAPoints[5]);
}

// Print out summary of percentages correct, incorrect, and none of the above, and the average number of classes over
// the p value threshold
/*
void printSummary() {
	std::cout << "\n--------------------------" << std::endl;
	std::cout << "\nSummary Statistics:\n";
	std::cout << "\n--------------------------\n";

	printPValueThreshold();

	int upperExtension = HYPERSPACE_MAX_BBOX_EXTENSION * 100;
	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		int lowerExtension = HYPERSPACE_BBOX_LOWER_BOUNDS[i] * 100;
		NOTACategory NOTALoc = static_cast<NOTACategory>(i);

		std::cout << "\n--------------------------" << std::endl;
		std::cout << "\nHyperspace NOTA Points (Lower BBOX Extension: " << lowerExtension <<
			"%, Upper BBOX Extension: " << upperExtension << "%):\n";
		std::cout << "\n--------------------------\n";

		printPredCategories(TEST_RESULTS.overallHyperspacePredStats[NOTALoc]);
		printPredCategoriesAllClasses(TEST_RESULTS.hyperspaceConfusionMatrix[NOTALoc], TEST_RESULTS.hyperspacePrecision[NOTALoc]);
		printNOTAStatistics(TEST_RESULTS.hyperspaceRandomPoints[NOTALoc]);
	}

	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		std::cout << "\n--------------------------" << std::endl;
		std::cout << "\nVoid NOTA Points (Nearest Neighbor Multiplier: " << NNStepMultiplier(i) << "):\n";
		std::cout << "\n--------------------------\n";
		printPredCategories(TEST_RESULTS.overallVoidPredStats[i]);
		printPredCategoriesAllClasses(TEST_RESULTS.voidConfusionMatrix[i], TEST_RESULTS.voidPrecision[i]);
		printNOTAStatistics(TEST_RESULTS.voidRandomPoints[i]);
	}
}
*/

void printSummary() {
	std::cout << "\n--------------------------" << std::endl;
	std::cout << "\nSummary Statistics:\n";
	std::cout << "\n--------------------------\n";

	for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
		printPValueThreshold(pvalCat);

		int upperExtension = HYPERSPACE_MAX_BBOX_EXTENSION * 100;
		for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
			int lowerExtension = HYPERSPACE_BBOX_LOWER_BOUNDS[i] * 100;
			NOTACategory NOTALoc = static_cast<NOTACategory>(i);

			std::cout << "\n--------------------------" << std::endl;
			std::cout << "\nHyperspace NOTA Points (Lower BBOX Extension: " << lowerExtension <<
				"%, Upper BBOX Extension: " << upperExtension << "%):\n";
			std::cout << "\n--------------------------\n";

			printPredCategories(TEST_RESULTS.overallHyperspacePredStats[pvalCat][NOTALoc]);
			printPredCategoriesAllClasses(TEST_RESULTS.hyperspaceConfusionMatrix[pvalCat][NOTALoc], TEST_RESULTS.hyperspacePrecision[pvalCat][NOTALoc]);
			printNOTAStatistics(TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTALoc]);
		}

		for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
			std::cout << "\n--------------------------" << std::endl;
			std::cout << "\nVoid NOTA Points (Nearest Neighbor Multiplier: " << NNStepMultiplier(i) << "):\n";
			std::cout << "\n--------------------------\n";
			printPredCategories(TEST_RESULTS.overallVoidPredStats[pvalCat][i]);
			printPredCategoriesAllClasses(TEST_RESULTS.voidConfusionMatrix[pvalCat][i], TEST_RESULTS.voidPrecision[pvalCat][i]);
			printNOTAStatistics(TEST_RESULTS.voidRandomPoints[pvalCat][i]);
		}
	}
}

/*
void cacheNNDistances(const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold) {
	FILE* fp = fopen(("nn-distances-iter" + std::to_string(fold) + ".csv").c_str(), "w");
	char delim;
	size_t testSize = nnDistances.size();

	delim = ',';
	for (const auto& iter = nnDistances[0].begin(); iter != nnDistances[0].end(); ++iter) {
		if (iter == nnDistances[0].end() - 1) {
			delim = '\n';
		}
		fprintf(fp, "%s%c", iter->first, delim);
	}

	for (size_t i = 0; i < testSize; ++i) {
		delim = ',';
		for (const auto& iter = nnDistances[i].begin(); iter != nnDistances[i].end(); ++iter) {
			if (iter == nnDistances[i].end() - 1) {
				delim = '\n';
			}
			fprintf(fp, "%g%c", iter->second, delim);
		}
	}

	fclose(fp);
}
*/

void test(std::vector<ClassMember>& dataset, size_t fold) {

	/*
	// Beginning of saving test datapoints before NOTA added and standardization
	std::vector<std::string> header = { "className" };
	size_t dim = dataset[0].features.size();
	for (size_t i = 0; i < dim; ++i) {
		header.push_back("col" + std::to_string(i));
	}

	std::vector<std::vector<std::string> > rows;
	for (const auto& obj : dataset) {
		std::vector<std::string> row = { obj.name };
		std::for_each(obj.features.begin(), obj.features.end(), [&row](double val) {
			row.push_back(std::to_string(val));
			});
		rows.push_back(row);
	}

	writeToCSV(header, rows, "iter" + std::to_string(fold) + "-test-datapoints.csv");
	// Ending of saving test datapoints before NOTA added and standardization
	*/	

	std::vector<ClassMember> standardizedDataset = standardize(dataset);

	/*
	// Beginning of saving test datapoints
	std::vector<std::string> header = { "className" };
	size_t dim = standardizedDataset[0].features.size();
	for (size_t i = 0; i < dim; ++i) {
		header.push_back("col" + std::to_string(i));
	}

	std::vector<std::vector<std::string> > rows;
	for (const auto& obj : standardizedDataset) {
		std::vector<std::string> row = { obj.name };
		std::for_each(obj.features.begin(), obj.features.end(), [&row](double val) {
			row.push_back(std::to_string(val));
			});
		rows.push_back(row);
	}

	writeToCSV(header, rows, "iter" + std::to_string(fold) + "-test-datapoints.csv");
	// Ending of saving test datapoints
	*/

	if (MODEL_STATE.processType == "PCA") {
		toPCASubspace(standardizedDataset);
	}

	// Find the nearest neighbor distance to each class
	std::vector<std::unordered_map<std::string, double> > nnDistances(standardizedDataset.size());

	size_t datapointsPerThread = std::round(standardizedDataset.size() / NUM_THREADS);
	std::vector<std::thread> threads;
	size_t start = 0, stop = 0;

	for (size_t i = 0; i < NUM_THREADS && stop != standardizedDataset.size(); ++i) {
		if (i == NUM_THREADS - 1) {
			stop = standardizedDataset.size();
		}
		else {
			stop = std::min(standardizedDataset.size(), stop + datapointsPerThread);
		}

		threads.emplace_back(std::thread{ computeClassDistances, std::cref(standardizedDataset), std::ref(nnDistances), start, stop });

		start = stop;
	}

	for (auto& t : threads) {
		t.join();
	}
	
	// Calculate the p value for each class
	std::vector<std::unordered_map<std::string, double> > pvalues(nnDistances.size());
	start = 0; stop = 0;

	for (size_t i = 0; i < NUM_THREADS && stop != nnDistances.size(); ++i) {
		if (i == NUM_THREADS - 1) {
			stop = nnDistances.size();
		}
		else {
			stop = std::min(nnDistances.size(), stop + datapointsPerThread);
		}

		threads[i] = std::thread{ calculatePValuesInRange, std::cref(nnDistances), std::ref(pvalues), start, stop };

		start = stop;
	}

	for (auto& t : threads) {
		t.join();
	}
	
	if (!MODEL_STATE.preexistingBestfit) {
		writeBestFitFunctionsToCSV(CACHE_PATHS.bestFitFunctionsByFoldFilepath, fold);
	}

	printPValues(dataset, pvalues);
	writePValuesToCSV(dataset, pvalues, fold);

	// Calculate the percentage of datapoints predicted correctly, incorrectly, or as none of the above
	std::vector<std::pair<std::string, std::string> > classifications = calculateStatistics(dataset, pvalues);

	//cacheTestPlotInfo(classifications, nnDistances, fold);

	// Print out statistics if this is the last fold
	if (fold == K_FOLDS - 1) {
		calculateSummary();
		writeNOTACategoryResultsToCSV();
		printSummary();
	}
}

void test(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold) {
	std::vector<std::unordered_map<std::string, double> > pvalues = calculatePValues(nnDistances);
	std::vector<std::pair<std::string, std::string> > classifications = calculateStatistics(dataset, pvalues);

	writeBestFitFunctionsToCSV(CACHE_PATHS.bestFitFunctionsByFoldFilepath, fold);
	writePValuesToCSV(dataset, pvalues, fold);
	cacheTestPlotInfo(classifications, nnDistances, fold);

	if (fold == K_FOLDS - 1) {
		calculateSummary();
		printSummary();
	}
}

void test(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold) {
	std::vector<std::pair<std::string, std::string> > classifications = calculateStatistics(dataset, pvalues);
	
	cacheTestPlotInfo(classifications, nnDistances, fold);
	
	if (fold == K_FOLDS - 1) {
		calculateSummary();
		printSummary();
	}
}
