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

std::unordered_map<std::string, double> createPerClassPValueThresholds() {
	std::unordered_map<std::string, double> pvalueThresholds;

	for (const auto& pair : MODEL_STATE.numInstancesPerClass) {
		pvalueThresholds[pair.first] = 1 / pair.second;
	}

	return pvalueThresholds;
}

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

void updateFeatureBB(const std::vector<double>& features) {
	size_t dim = features.size();
	
	for (size_t i = 0; i < dim; ++i) {
		if (features[i] < MODEL_STATE.featureMins[i]) {
			MODEL_STATE.featureMins[i] = features[i];
		}
		if (features[i] > MODEL_STATE.featureMaxs[i]) {
			MODEL_STATE.featureMaxs[i] = features[i];
		}
	}
}

void findFeatureBB(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset, double extension) {
	MODEL_STATE.featureMins = MODEL_STATE.featureMaxs = dataset.begin()->second[0].features;
	size_t dim = dataset.begin()->second[0].features.size();

	for (const auto& pair : dataset) {
		for (const auto& obj : pair.second) {
			updateFeatureBB(obj.features);
		}
	}

	for (size_t i = 0; i < dim; ++i) {
		double extensionVal = (MODEL_STATE.featureMaxs[i] - MODEL_STATE.featureMins[i]) * extension;
		MODEL_STATE.featureMins[i] -= extensionVal;
		MODEL_STATE.featureMaxs[i] += extensionVal;
	}
}

void insertRandomPoints(std::vector<ClassMember>& dataset, size_t numInsert, size_t iteration) {
	size_t total = dataset.size();
	size_t totalAfterInsert = total + numInsert;
	size_t dim = MODEL_STATE.featureMins.size();
	dataset.reserve(totalAfterInsert);
	dataset.insert(dataset.end(), numInsert, ClassMember(std::vector<double>(dim), "", 0, true));
	std::random_device rd;
	std::mt19937 gen(rd());
	
	for (size_t i = 0; i < dim; ++i) {
		std::uniform_real_distribution<double> unif(MODEL_STATE.featureMins[i], MODEL_STATE.featureMaxs[i]);
		for (size_t j = total; j < totalAfterInsert; ++j) {
			dataset[j].features[i] = unif(gen);
		}
	}

	/*
	// Beginning of saving NOTA points
	std::string filename = "iter" + std::to_string(iteration) + "-NOTA.csv";
	char delim;
	FILE* fp = fopen(filename.c_str(), "w");
	for (size_t i = 0; i < dim; ++i) {
		delim = (i == dim - 1) ? '\n' : ',';
		fprintf(fp, "col%lu%c", i, delim);
	}
	for (size_t i = total; i < totalAfterNOTA; ++i) {
		for (size_t j = 0; j < dim; ++j) {
			delim = (j == dim - 1) ? '\n' : ',';
			fprintf(fp, "%g%c", dataset[i].features[j], delim);
		}
	}
	fclose(fp);
	// Ending of saving NOTA points
	*/
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
		std::string expectedClass = (dataset[i].NOTA) ? "Injected \"none of the above\" point" : dataset[i].name;
		std::cout << "Actual class: " << expectedClass << std::endl;
	}
}

void writeBestFitFunctionsToCSV(size_t fold) {
	FILE* fp;

	if (fold == 0) {
		fp = fopen(CACHE_PATHS.bestFitFunctionsFilepath.c_str(), "w");
		fprintf(fp, "fold,class,bestFitFunction,c,a,residual\n");
	}
	else {
		fp = fopen(CACHE_PATHS.bestFitFunctionsFilepath.c_str(), "a");
	}

	for (const auto& pair : MODEL_STATE.bestFit) {
		fprintf(fp, "%lu,%s,%s,%g,%g,%g\n", fold, pair.first.c_str(), pair.second.functionName.c_str(),
			pair.second.c[0], pair.second.c[1], pair.second.wrmsError);
	}

	fclose(fp);
}

void writePValuesToCSV(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues, size_t fold) {
	FILE* fp;
	char delim;
	size_t totalDatapoints = pvalues.size();
	size_t totalClasses = MODEL_STATE.classNames.size();

	// Create a header if this is the first time writing to the CSV file
	if (fold == 0) {
		fp = fopen(CACHE_PATHS.pvaluesFilepath.c_str(), "w");
		fprintf(fp, "lineNumber,fold,");
		for (size_t i = 0; i < totalClasses; ++i) {
			delim = (i == totalClasses - 1) ? '\n' : ',';
			fprintf(fp, "%s%c", MODEL_STATE.classNames[i].c_str(), delim);
		}
	}
	else {
		fp = fopen(CACHE_PATHS.pvaluesFilepath.c_str(), "a");
	}

	for (size_t i = 0; i < totalDatapoints; ++i) {
		fprintf(fp, "%lu,%lu,", dataset[i].lineNumber, fold);
		for (size_t j = 0; j < totalClasses; ++j) {
			delim = (j == totalClasses - 1) ? '\n' : ',';
			std::string className = MODEL_STATE.classNames[j];
			if (pvalues[i].find(className) != pvalues[i].end()) {
				fprintf(fp, "%g%c", pvalues[i].at(className), delim);
			}
			else {
				fprintf(fp, "0%c", delim);
			}
		}
	}

	fclose(fp);
}

/*
void calculateStatistics(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::vector<double>& pvalueThresholds) {
	size_t total = dataset.size(), numThresholds = pvalueThresholds.size();

	for (size_t i = 0; i < total; ++i) {
		const std::string& expectedClass = dataset[i].name;

		// Find the largest p value for the datapoint
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		for (size_t j = 0; j < numThresholds; ++j) {
			std::string threshold;
			if (TEST_RESULTS.pvalueThreshold < 0 && j < TOTAL_DYNAMIC_PVALUES) {
				threshold = std::string("1/") + std::to_string(std::max(PVALUE_INCREMENT * j, 1.0)) + "n";
			}
			else {
				threshold = std::to_string(pvalueThresholds[j]);
			}

			// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA) for overall statistics
			// Tally whether p value indicates a true positive, false positive, or false negative for a class
			if (largestPValue->second >= pvalueThresholds[j]) {
				if (dataset[i].NOTA) {
					++TEST_RESULTS.confusionMatrix[threshold + "|" + largestPValue->first][2];
					++TEST_RESULTS.injectedNOTA[2];
				}
				else if (largestPValue->first == expectedClass) {
					++TEST_RESULTS.overallPredStats[threshold][2];
					++TEST_RESULTS.confusionMatrix[threshold + "|" + largestPValue->first][0];
				}
				else {
					++TEST_RESULTS.overallPredStats[threshold][3];
					++TEST_RESULTS.confusionMatrix[threshold + "|" + largestPValue->first][2];
					++TEST_RESULTS.confusionMatrix[threshold + "|" + expectedClass][1];
				}
			}
			else {
				if (dataset[i].NOTA) {
					++TEST_RESULTS.injectedNOTA[1];
				}
				else {
					++TEST_RESULTS.overallPredStats[threshold][4];
					++TEST_RESULTS.confusionMatrix[threshold + "|" + expectedClass][1];
					++TEST_RESULTS.injectedNOTA[3];
				}
			}

			// Count how many classes were over the p value threshold
			for (const auto& pair : pvalues[i])
				if (pair.second >= pvalueThresholds[j]) ++TEST_RESULTS.overallPredStats[threshold][1];
		}
	}
}
*/

std::vector<std::pair<std::string, std::string> > calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues, double pvalueThreshold) {
	size_t total = dataset.size();
	std::vector<std::pair<std::string, std::string> > classifications(total);

	for (size_t i = 0; i < total; ++i) {
		const std::string& expectedClass = dataset[i].name;

		// Find the largest p value for the datapoint
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA) for overall statistics
		// Tally whether p value indicates a true positive, false positive, or false negative for a class
		if (largestPValue->second >= pvalueThreshold) {
			if (dataset[i].NOTA) {
				++TEST_RESULTS.confusionMatrix[largestPValue->first][2];
				++TEST_RESULTS.randomPoints[2];
			}
			else if (largestPValue->first == expectedClass) {
				++TEST_RESULTS.overallPredStats[0];
				++TEST_RESULTS.confusionMatrix[largestPValue->first][0];
			}
			else {
				++TEST_RESULTS.overallPredStats[1];
				++TEST_RESULTS.confusionMatrix[largestPValue->first][2];
				++TEST_RESULTS.confusionMatrix[expectedClass][1];
			}
			classifications[i] = { (dataset[i].NOTA) ? "NOTA" : expectedClass, largestPValue->first};
		}
		else {
			if (dataset[i].NOTA) {
				++TEST_RESULTS.randomPoints[1];
			}
			else {
				++TEST_RESULTS.overallPredStats[2];
				++TEST_RESULTS.confusionMatrix[expectedClass][1];
				++TEST_RESULTS.randomPoints[3];
			}
			classifications[i] = { (dataset[i].NOTA) ? "NOTA" : expectedClass, "NOTA" };
		}
	}

	return classifications;
}

std::vector<std::pair<std::string, std::string> > calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::unordered_map<std::string, double>& pvalueThresholds) {
	size_t total = dataset.size();
	std::vector<std::pair<std::string, std::string> > classifications(total);

	for (size_t i = 0; i < total; ++i) {
		const std::string& expectedClass = dataset[i].name;
		
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		if (largestPValue->second >= pvalueThresholds.at(largestPValue->first)) {
			if (dataset[i].NOTA) {
				++TEST_RESULTS.confusionMatrix[largestPValue->first][2];
				++TEST_RESULTS.randomPoints[2];
			}
			else if (largestPValue->first == expectedClass) {
				++TEST_RESULTS.overallPredStats[0];
				++TEST_RESULTS.confusionMatrix[largestPValue->first][0];
			}
			else {
				++TEST_RESULTS.overallPredStats[1];
				++TEST_RESULTS.confusionMatrix[largestPValue->first][2];
				++TEST_RESULTS.confusionMatrix[expectedClass][1];
			}
			classifications[i] = { (dataset[i].NOTA) ? "NOTA" : expectedClass, largestPValue->first };
		}
		else {
			if (dataset[i].NOTA) {
				++TEST_RESULTS.randomPoints[1];
			}
			else {
				++TEST_RESULTS.overallPredStats[2];
				++TEST_RESULTS.confusionMatrix[expectedClass][1];
				++TEST_RESULTS.randomPoints[3];
			}
			classifications[i] = { (dataset[i].NOTA) ? "NOTA" : expectedClass, "NOTA" };
		}
	}

	return classifications;
}

std::vector<std::pair<std::string, std::string> > calculateStatistics(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues) {
	std::vector<std::pair<std::string, std::string> > classifications;

	if (TEST_RESULTS.pvalueThreshold >= 0) {
		classifications = calculateStatistics(dataset, pvalues, TEST_RESULTS.pvalueThreshold);
	}
	else {
		classifications = calculateStatistics(dataset, pvalues, TEST_RESULTS.perClassThresholds);
	}

	return classifications;
}

void calculateSummary() {
	// Determine size of entire dataset used for training/testing plus "randomly placed" points
	size_t total = MODEL_STATE.datasetSize + TEST_RESULTS.randomPoints[0];

	for (const std::string& className : MODEL_STATE.classNames) {
		// Calculate number of true negatives for each class using the current p value threshold
		double* classConfusionMatrix = TEST_RESULTS.confusionMatrix[className];
		classConfusionMatrix[3] = total - classConfusionMatrix[0] - classConfusionMatrix[1] - classConfusionMatrix[2];

		// Calculate the precision for each class using the current p value threshold
		TEST_RESULTS.precision[className] = (classConfusionMatrix[0] + classConfusionMatrix[2] == 0) ?
			0 : (classConfusionMatrix[0] / (classConfusionMatrix[0] + classConfusionMatrix[2])) * 100;
	}

	for (size_t i = 0; i < 3; ++i) {
		TEST_RESULTS.overallPredStats[i] = (TEST_RESULTS.overallPredStats[i] / MODEL_STATE.datasetSize) * 100;
	}

	TEST_RESULTS.randomPoints[4] = total - TEST_RESULTS.randomPoints[1] - TEST_RESULTS.randomPoints[2] -
		TEST_RESULTS.randomPoints[3];
	TEST_RESULTS.randomPoints[5] = (TEST_RESULTS.randomPoints[1] /
		(TEST_RESULTS.randomPoints[1] + TEST_RESULTS.randomPoints[2])) * 100;
}

void printPValueThreshold() {
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

void printPredCategories() {
	int defaultPrecision = std::cout.precision();
	std::cout << std::setprecision(2) << std::fixed;
	std::cout << "Correct: " << TEST_RESULTS.overallPredStats[0] << "%\n";
	std::cout << "Incorrect: " << TEST_RESULTS.overallPredStats[1] << "%\n";
	std::cout << "None of the above: " << TEST_RESULTS.overallPredStats[2] << "%\n\n";
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

void printPredCategoriesPerClass(const std::string& className) {
	printConfusionMatrix(TEST_RESULTS.confusionMatrix.at(className));

	// Print out precision for class
	int defaultPrecision = std::cout.precision();
	std::cout << "Precision: " << std::setprecision(2) << std::fixed << TEST_RESULTS.precision[className] << "%\n\n";
	//std::cout << "Precision: " << std::setprecision(2) << std::fixed << TEST_RESULTS.precision[classKey] << "%\n";
	std::cout << std::setprecision(defaultPrecision);
	std::cout.unsetf(std::ios::fixed);
}

void printPredCategoriesAllClasses() {
	for (const std::string& className : MODEL_STATE.classNames) {
		std::cout << "Class: " << className << std::endl;
		std::cout << "Total instances: " << MODEL_STATE.numInstancesPerClass.at(className) << std::endl;
		printPredCategoriesPerClass(className);
	}
}

void printNOTAStatistics() {
	std::cout << "\"None of the above\" classifications:\n";
	std::cout << "Total randomly placed points: " << TEST_RESULTS.randomPoints[0] << std::endl;
	printConfusionMatrix(&TEST_RESULTS.randomPoints[1]);
	printf("Recall: %.2f%%\n\n", TEST_RESULTS.randomPoints[5]);
}

// Print out summary of percentages correct, incorrect, and none of the above, and the average number of classes over
// the p value threshold
void printSummary() {
	std::cout << "\n--------------------------" << std::endl;
	std::cout << "\nSummary Statistics:\n";
	std::cout << "\n--------------------------\n";
	printPValueThreshold();
	printPredCategories();
	printPredCategoriesAllClasses();
	printNOTAStatistics();
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

void cacheTestPlotInfo(const std::vector<std::pair<std::string, std::string> >& classifications,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold) {
	if (fold == 0) {
		createFolder(CACHE_PATHS.classificationsDirectory.c_str());
	}
	std::string filepath = CACHE_PATHS.classificationsDirectory + getPathSep() + "classifications-iter" +
		std::to_string(fold) + ".csv";
	FILE* fp = fopen(filepath.c_str(), "w");
	size_t testSize = nnDistances.size();

	fprintf(fp, "className,classification");
	for (auto iter = nnDistances[0].begin(); iter != nnDistances[0].end(); ++iter) {
		fprintf(fp, ",%s", iter->first.c_str());
	}
	fprintf(fp, "\n");

	for (size_t i = 0; i < testSize; ++i) {
		fprintf(fp, "%s,%s", classifications[i].first.c_str(), classifications[i].second.c_str());
		for (auto iter = nnDistances[i].begin(); iter != nnDistances[i].end(); ++iter) {
			fprintf(fp, ",%g", iter->second);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}

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
	
	writeBestFitFunctionsToCSV(fold);

	/*
	std::cout << "P Value Thresholds Per Class:\n";
	for (const auto& pair : MODEL_STATE.classMap) {
		std::cout << pair.first << ": " << pvalueThresholds[pair.first] << std::endl;
	}
	*/

	printPValues(dataset, pvalues);
	writePValuesToCSV(dataset, pvalues, fold);

	// Calculate the percentage of datapoints predicted correctly, incorrectly, or as none of the above
	std::vector<std::pair<std::string, std::string> > classifications = calculateStatistics(dataset, pvalues);

	cacheTestPlotInfo(classifications, nnDistances, fold);

	/*
	for (const auto& pair : TEST_RESULTS.overallPredStats) {
		std::cout << "Threshold: " << pair.first << std::endl;
		std::cout << "Correct count: " << pair.second[2] << std::endl;
		std::cout << "Incorrect count: " << pair.second[3] << std::endl;
		std::cout << "NOTA count: " << pair.second[4] << std::endl;
	}

	for (const auto& pair : TEST_RESULTS.confusionMatrix) {
		std::cout << "Threshold: " << pair.first << std::endl;
		std::cout << "TP: " << pair.second[0] << std::endl;
		std::cout << "FN: " << pair.second[1] << std::endl;
		std::cout << "FP: " << pair.second[2] << std::endl;
	}

	std::cout << "Total injected NOTA points: " << TEST_RESULTS.injectedNOTA[0] << std::endl;
	std::cout << "TP: " << TEST_RESULTS.injectedNOTA[1] << std::endl;
	std::cout << "FN: " << TEST_RESULTS.injectedNOTA[2] << std::endl;
	std::cout << "FP: " << TEST_RESULTS.injectedNOTA[3] << std::endl;
	*/

	// Print out statistics if this is the last fold
	if (fold == K_FOLDS - 1) {
		calculateSummary();
		printSummary();
	}
}

void test(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold) {
	std::vector<std::unordered_map<std::string, double> > pvalues = calculatePValues(nnDistances);
	std::vector<std::pair<std::string, std::string> > classifications = calculateStatistics(dataset, pvalues);

	writeBestFitFunctionsToCSV(fold);
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
