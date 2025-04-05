#include <cstdlib>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <iterator>
#include <thread>
#include <cassert>
#include <ap.h>

#include "classMember.h"
#include "process.h"
#include "modelState.h"
#include "fit.h"
#include "test.h"

// Helper function to fill the k sets to a certain amount per set
void fillFoldToLimit(std::uniform_int_distribution<>& distrib, std::mt19937& gen, int fillLimit,
	std::vector<ClassMember>& dataset, size_t start, std::vector<ClassMember> kSets[]) {
	bool foldFull[K_FOLDS] = { false };

	size_t foldOriginalSizes[K_FOLDS];
	for (int i = 0; i < K_FOLDS; ++i) {
		foldOriginalSizes[i] = kSets[i].size();
	}

	int iSet;
	for (size_t iData = start; iData < std::min(start + fillLimit * K_FOLDS, dataset.size()); ++iData) {
		do {
			iSet = distrib(gen);
		} while (foldFull[iSet]);

		kSets[iSet].push_back(std::move(dataset[iData]));

		if (kSets[iSet].size() == foldOriginalSizes[iSet] + fillLimit) {
			foldFull[iSet] = true;
		}
	}
}

// Split dataset into k sets
void kFoldSplit(std::vector<ClassMember>& dataset, std::vector<ClassMember> kSets[]) {
	// Create random number generator
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distrib(0, K_FOLDS - 1);

	int minElems = dataset.size() / K_FOLDS;

	// Evenly divide largest multiple of K_FOLDS possible among the k sets
	fillFoldToLimit(distrib, gen, minElems, dataset, 0, kSets);

	// Place remainder into the sets with each set receiving at most one extra datapoint
	fillFoldToLimit(distrib, gen, 1, dataset, minElems * K_FOLDS, kSets);
}

// Create default p value thresholds from constants and 1/(m * n) where n is the number of datapoints in the largest class
// and m is some multiplier
std::vector<double> createPValueThresholds(std::unordered_map<std::string, double[5]>& predictionStatistics) {
	std::vector<double> pvalueThresholds(TOTAL_DYNAMIC_PVALUES);
	
	// Find the class with the most datapoints
	using pairtype = decltype(MODEL_STATE.classMap)::value_type;
	auto largestClass = std::max_element(MODEL_STATE.classMap.begin(), MODEL_STATE.classMap.end(),
		[](const pairtype& p1, const pairtype& p2) {
			return p1.second.size() < p2.second.size();
		});
	int n = largestClass->second.size();

	// Calculate p value thresholds and store them in predictionStatistics
	std::string thresholdName;
	double pvalueThreshold, multiplier;

	for (int i = 0; i < TOTAL_DYNAMIC_PVALUES; ++i) {
		multiplier = std::max(PVALUE_INCREMENT * i, 1.0);
		thresholdName = std::string("1/") + std::to_string(multiplier) + "n";
		pvalueThreshold = 1 / (multiplier * n);
		pvalueThresholds[i] = pvalueThreshold;
		predictionStatistics[thresholdName][0] += pvalueThreshold;
	}

	// Insert constant p value thresholds
	pvalueThresholds.insert(pvalueThresholds.end(), CONSTANT_PVALUE_THRESHOLDS.begin(), CONSTANT_PVALUE_THRESHOLDS.end());
	for (double constPValue : CONSTANT_PVALUE_THRESHOLDS) {
		predictionStatistics[std::to_string(constPValue)][0] += constPValue;
	}

	// At least one default p value threshold must exist
	assert(!pvalueThresholds.empty());

	return pvalueThresholds;
}

// Normalize the feature values of the test set with the means and sigmas calculated during training
std::vector<ClassMember> normalize(std::vector<ClassMember> dataset) {
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
		double minDistance = std::numeric_limits<double>::max();
		std::unordered_map<std::string, double> classDistance;
		for (const auto& pair : MODEL_STATE.classMap) {
			for (const auto& classDatapoint : pair.second) {
				double distance = euclideanDistance(dataset[i].features, classDatapoint.features);
				if (distance < minDistance) {
					minDistance = distance;
				}
			}
			classDistance[pair.first] = minDistance;
		}
		m.lock();
		nnDistances[i] = std::move(classDistance);
		m.unlock();
	}
}

// Calculate the p values of the datapoints in the test set for each class
void calculatePValues(const std::vector<std::unordered_map<std::string, double> >& dataset,
	std::vector<std::unordered_map<std::string, double> >& pvalues, size_t start, size_t stop) {
	for (size_t i = start; i < stop; ++i) {
		std::unordered_map<std::string, double> result;
		for (const auto& pair : dataset[i]) {
			std::string bestFitFunction = MODEL_STATE.bestFit[pair.first].functionName;
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
		std::cout << "Actual class: " << dataset[i].name << std::endl;
	}
}

// Calculate the percentage predicted correctly, incorrectly, and as none of the above, and the average number of classes per
// datapoint that are over the p value threshold
void calculateStatistics(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::vector<double>& pvalueThresholds, std::unordered_map<std::string, double[5]>& predictionStatistics, bool userPValueThreshold) {
	size_t total = dataset.size(), numThresholds = pvalueThresholds.size();
	std::vector<double> numCorrect(numThresholds, 0), numIncorrect(numThresholds, 0), numNOTA(numThresholds, 0),
		classesOverThreshold(numThresholds, 0);

	for (size_t i = 0; i < total; ++i) {
		// Find the largest p value for the datapoint
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		for (size_t j = 0; j < numThresholds; ++j) {
			// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA)
			if (largestPValue->second >= pvalueThresholds[j]) {
				if (largestPValue->first == dataset[i].name) { ++numCorrect[j]; }
				else { ++numIncorrect[j]; }
			}
			else { ++numNOTA[j]; }

			// Count how many classes were over the p value threshold
			for (const auto& pair : pvalues[i])
				if (pair.second >= pvalueThresholds[j]) ++classesOverThreshold[j];
		}
	}

	/* Add percentages for correct, incorrect, and NOTA predictions, and the average number of classes per datapoint
	with a p value over the threshold. */
	for (size_t i = 0; i < numThresholds; ++i) {
		std::string threshold;
		if (!userPValueThreshold && i < TOTAL_DYNAMIC_PVALUES) {
			threshold = std::string("1/") + std::to_string(std::max(PVALUE_INCREMENT * i, 1.0)) + "n";
		}
		else {
			threshold = std::to_string(pvalueThresholds[i]);
		}
		predictionStatistics[threshold][1] += classesOverThreshold[i] / total;
		predictionStatistics[threshold][2] += (numCorrect[i] / total) * 100;
		predictionStatistics[threshold][3] += (numIncorrect[i] / total) * 100;
		predictionStatistics[threshold][4] += (numNOTA[i] / total) * 100;
	}
}

// Calculate the average p value, classes over the p value threshold, correct, incorrect, and none of the above across k folds
void calculateSummary(std::unordered_map<std::string, double[5]>& predictionStatistics) {
	for (auto& pair : predictionStatistics) {
		for (int i = 0; i < 5; ++i)
			pair.second[i] /= K_FOLDS;
	}
}

void printPredCategories(const double results[]) {
	int defaultPrecision = std::cout.precision();

	std::cout << "Average number of classes over the p value threshold per datapoint: " << results[1] << std::endl;
	std::cout << std::setprecision(2) << std::fixed;
	std::cout << "Correct: " << results[2] << "%\n";
	std::cout << "Incorrect: " << results[3] << "%\n";
	std::cout << "None of the above: " << results[4] << "%\n";
	std::cout << std::setprecision(defaultPrecision);
	std::cout.unsetf(std::ios::fixed);
}

// Print out summary of percentages correct, incorrect, and none of the above, and the average number of classes over
// the p value threshold when using p value threshold specified by the user
void printSummary(const double statistics[]) {
	std::cout << "\nSummary Statistics:\n";
	printPredCategories(statistics);
}

// Print out summary of percentages correct, incorrect, and none of the above, and the average number of classes over
// the p value threshold when using default p value thresholds
void printSummary(const std::unordered_map<std::string, double[5]>& predictionStatistics) {
	std::cout << "\nSummary Statistics (n is the greatest number of datapoints belonging to a class):\n";

	std::string threshold;
	for (int i = TOTAL_DYNAMIC_PVALUES - 1; i >= 0; --i) {
		threshold = std::string("1/") + std::to_string(std::max(PVALUE_INCREMENT * i, 1.0)) + "n";
		std::cout << "Average p value threshold: " << threshold << " = " << predictionStatistics.at(threshold)[0] << std::endl;
		printPredCategories(predictionStatistics.at(threshold));
	}

	std::vector<double> constantThresholds = CONSTANT_PVALUE_THRESHOLDS;
	std::sort(constantThresholds.begin(), constantThresholds.end());
	for (double constThreshold : constantThresholds) {
		std::cout << "Constant p value threshold: " << constThreshold << std::endl;
		printPredCategories(predictionStatistics.at(std::to_string(constThreshold)));
	}
}

void test(const std::vector<ClassMember>& dataset, std::unordered_map<std::string, double[5]>& predictionStatistics, size_t fold, double pvalueThreshold) {
	// Create p value thresholds if none are provided by the user
	std::vector<double> pvalueThresholds;
	bool userPValueThreshold;
	if (pvalueThreshold == std::numeric_limits<double>::lowest()) {
		pvalueThresholds = createPValueThresholds(predictionStatistics);
		userPValueThreshold = false;
	}
	else {
		pvalueThresholds.push_back(pvalueThreshold);
		predictionStatistics[std::to_string(pvalueThreshold)][0] += pvalueThreshold;
		userPValueThreshold = true;
	}
	
	std::vector<ClassMember> normalizedDataset = normalize(dataset);

	toPCASubspace(normalizedDataset);

	// Find the nearest neighbor distance to each class
	std::vector<std::unordered_map<std::string, double> > nnDistances(normalizedDataset.size());

	size_t datapointsPerThread = std::round(normalizedDataset.size() / NUM_THREADS);
	std::vector<std::thread> threads;
	size_t i = 0, start = 0, stop = 0;

	while (i < NUM_THREADS && stop != normalizedDataset.size()) {
		if (i == NUM_THREADS - 1) {
			stop = normalizedDataset.size();
		}
		else {
			stop = std::min(normalizedDataset.size(), stop + datapointsPerThread);
		}

		threads.emplace_back(std::thread{ computeClassDistances, std::cref(normalizedDataset), std::ref(nnDistances), start, stop });

		++i;
		start = stop;
	}

	for (auto& t : threads) {
		t.join();
	}

	// Calculate the p value for each class
	std::vector<std::unordered_map<std::string, double> > pvalues(nnDistances.size());
	threads.clear();
	i = 0, start = 0, stop = 0;

	while (i < NUM_THREADS && stop != nnDistances.size()) {
		if (i == NUM_THREADS - 1) {
			stop = nnDistances.size();
		}
		else {
			stop = std::min(nnDistances.size(), stop + datapointsPerThread);
		}

		threads[i] = std::thread{ calculatePValues, std::cref(nnDistances), std::ref(pvalues), start, stop };

		++i;
		start = stop;
	}

	for (auto& t : threads) {
		t.join();
	}

	// Print out the p values for each datapoint
	printPValues(dataset, pvalues);

	// Calculate the percentage of datapoints predicted correctly, incorrectly, or as none of the above
	calculateStatistics(dataset, pvalues, pvalueThresholds, predictionStatistics, userPValueThreshold);

	// Print out statistics if this is the last fold
	if (fold == K_FOLDS - 1) {
		calculateSummary(predictionStatistics);
		if (userPValueThreshold) {
			printSummary(predictionStatistics[std::to_string(pvalueThreshold)]);
		}
		else {
			printSummary(predictionStatistics);
		}
	}
}