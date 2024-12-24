#include <cstdlib>
#include <ctime>
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

void trainTestSplit(std::vector<ClassMember>& dataset, std::vector<ClassMember>& testDataset, double testSize) {
	// Seed random number generator based on the current time
	std::time_t timestamp;
	std::time(&timestamp);
	std::cout << "Randomness for train/test split seeded to " << timestamp << ".\n";

	std::srand(timestamp);
	int numTestElems = dataset.size() * testSize;
	for (int i = 0; i < numTestElems; ++i) {
		int transferElem = std::rand() % dataset.size();
		testDataset.push_back(std::move(dataset[transferElem]));
		dataset.erase(dataset.begin() + transferElem);
	}
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

//Print out the p values for every datapoint in the test set and some summary statistics
void printPredictionResults(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues, double pvalueThreshold) {
	double numCorrect = 0, numIncorrect = 0, numNOTA = 0, total = dataset.size();
	std::cout << "P values for test set:\n";
	for (size_t i = 0; i < dataset.size(); ++i) {
		// Print out the feature values for the datapoint
		std::cout << "Feature values: ";
		std::copy(dataset[i].features.begin(), dataset[i].features.end(), std::ostream_iterator<double>(std::cout, ", "));
		std::cout << std::endl;

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

		// Print out the p values of the datapoint for each class
		for (const auto& pair : pvalues[i]) {
			std::cout << pair.first << ": " << pair.second << std::endl;
		}

		// Print out the actual class of the datapoint
		std::cout << "Actual class: " << dataset[i].name << std::endl;
	}

	// Print out percentages for how many were predicted correctly, incorrectly, and as none of the above
	double percentCorrect = (numCorrect / total) * 100;
	double percentIncorrect = (numIncorrect / total) * 100;
	double percentNOTA = (numNOTA / total) * 100;

	std::cout << "\nSummary Statistics:\n";
	std::cout << std::setprecision(2) << std::fixed;
	std::cout << "Correct: " << percentCorrect << "%\n";
	std::cout << "Incorrect: " << percentIncorrect << "%\n";
	std::cout << "None of the above: " << percentNOTA << "%\n";
}

void test(const std::vector<ClassMember>& dataset, double pvalueThreshold) {
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

	printPredictionResults(dataset, pvalues, pvalueThreshold);
}