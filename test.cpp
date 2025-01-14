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

#include "classMember.h"
#include "process.h"
#include "modelState.h"
#include "fit.h"

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

// Normalize the feature values of the test set with the means and sigmas calculated during training
std::vector<ClassMember> normalize(std::vector<ClassMember> dataset) {
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

// Calculate the percentage predicted correctly, incorrectly, and as none of the above
std::vector<double> calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues, double pvalueThreshold) {
	double numCorrect = 0, numIncorrect = 0, numNOTA = 0, total = dataset.size();

	for (size_t i = 0; i < dataset.size(); ++i) {
		// Find the largest p value for the datapoint
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA)
		if (largestPValue->second >= pvalueThreshold) {
			if (largestPValue->first == dataset[i].name) { ++numCorrect; }
			else { ++numIncorrect; }
		}
		else { ++numNOTA; }
	}

	double percentCorrect = (numCorrect / total) * 100;
	double percentIncorrect = (numIncorrect / total) * 100;
	double percentNOTA = (numNOTA / total) * 100;

	return std::vector<double> {percentCorrect, percentIncorrect, percentNOTA};
}

// Calculate the average correct, incorrect, and none of the above across k folds
std::vector<double> calculateSummary() {
	double avgCorrect = 0, avgIncorrect = 0, avgNOTA = 0;

	for (size_t i = 0; i < K_FOLDS; ++i) {
		avgCorrect += predictionStatistics[i][0];
		avgIncorrect += predictionStatistics[i][1];
		avgNOTA += predictionStatistics[i][2];
	}

	return std::vector<double> {avgCorrect / K_FOLDS, avgIncorrect / K_FOLDS, avgNOTA / K_FOLDS};
}

// Print out summary of percentages correct, incorrect, and none of the above
void printSummary(double percentCorrect, double percentIncorrect, double percentNOTA) {
	std::cout << "\nSummary Statistics:\n";
	std::cout << std::setprecision(2) << std::fixed;
	std::cout << "Correct: " << percentCorrect << "%\n";
	std::cout << "Incorrect: " << percentIncorrect << "%\n";
	std::cout << "None of the above: " << percentNOTA << "%\n";
}

void test(const std::vector<ClassMember>& dataset, double pvalueThreshold, size_t fold) {
	std::vector<ClassMember> normalizedDataset = normalize(dataset);	
	
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
	std::vector<double> stats = calculateStatistics(dataset, pvalues, pvalueThreshold);
	for (size_t i = 0; i < 3; ++i) {
		predictionStatistics[fold][i] = stats[i];
	}

	// Print out statistics if this is the last fold
	if (fold == K_FOLDS - 1) {
		std::vector<double> summary = calculateSummary();
		printSummary(summary[0], summary[1], summary[2]);
	}
}