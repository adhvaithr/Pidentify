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
#include <set>
#include "ap.h"

#include "classMember.h"
#include "process.h"
#include "modelState.h"
#include "fit.h"
#include "test.h"
#include "CSVWrite.hpp"
#include "testResults.h"

// Create default p value thresholds from constants and 1/(m * n) where n is the number of datapoints in the largest class
// and m is some multiplier
std::vector<double> createPValueThresholds() {
	// Initialize vector of p value thresholds when there is user defined p value threshold
	if (TEST_RESULTS.pvalueThreshold >= 0) {
		TEST_RESULTS.overallPredStats[std::to_string(TEST_RESULTS.pvalueThreshold)][0] += TEST_RESULTS.pvalueThreshold;
		return { TEST_RESULTS.pvalueThreshold };
	}

	// Create default p value thresholds when there is no user defined p value threshold
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
		TEST_RESULTS.overallPredStats[thresholdName][0] += pvalueThreshold;
	}

	// Insert constant p value thresholds
	pvalueThresholds.insert(pvalueThresholds.end(), CONSTANT_PVALUE_THRESHOLDS.begin(), CONSTANT_PVALUE_THRESHOLDS.end());
	for (double constPValue : CONSTANT_PVALUE_THRESHOLDS) {
		TEST_RESULTS.overallPredStats[std::to_string(constPValue)][0] += constPValue;
	}

	// At least one default p value threshold must exist
	assert(!pvalueThresholds.empty());

	return pvalueThresholds;
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

void writeBestFitFunctionsToCSV(const std::string& filename, size_t fold) {
	std::vector<std::string> header;
	std::vector<std::vector<std::string> > rows;

	if (fold == 0) {
		std::string columnNames[] = { "fold", "class", "bestFitFunction", "c", "a", "residual" };
		for (std::string columnName : columnNames) {
			header.push_back(columnName);
		}
	}
		
	for (const auto& pair : MODEL_STATE.bestFit) {
		std::vector<std::string> row = { std::to_string(fold), pair.first, pair.second.functionName, std::to_string(pair.second.c[0]),
			std::to_string(pair.second.c[1]), std::to_string(pair.second.wrmsError) };
		rows.push_back(std::move(row));
	}

	writeToCSV(header, rows, filename);
}

// Write the p values for each datapoint in the test set to a CSV file
void writePValuesToCSV(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::string& filename, size_t fold) {
	size_t totalDatapoints = pvalues.size();
	std::vector<std::string> header, classNames(pvalues[0].size());
	std::vector<std::vector<double> > rows(totalDatapoints);
	std::transform(pvalues[0].begin(), pvalues[0].end(), classNames.begin(),
		[](const std::pair<std::string, double>& pair) {
			return pair.first;
		});
	std::sort(classNames.begin(), classNames.end());

	// Create a header if this is the first time writing to the CSV file
	if (fold == 0) {
		header.emplace_back("lineNumber");
		header.insert(header.begin() + 1, classNames.begin(), classNames.end());
	}
	
	for (size_t i = 0; i < totalDatapoints; ++i) {
		rows[i].push_back(dataset[i].lineNumber);
		for (const std::string& className : classNames) {
			rows[i].push_back(pvalues[i].at(className));
		}
	}

	writeToCSV(header, rows, filename);
}

void createOverallStatRow(const std::unordered_map<std::string, double[5]>& predictionStatistics,
	const std::string& threshold, std::vector<std::vector<std::string> >& rows, size_t currentRow) {
	std::stringstream ss;
	std::string entry;
	int defaultPrecision = ss.precision();
	ss << predictionStatistics.at(threshold)[0] << ' ' << predictionStatistics.at(threshold)[1] << ' ';
	ss << std::fixed << std::setprecision(2);
	ss << predictionStatistics.at(threshold)[2] << ' ' << predictionStatistics.at(threshold)[3] << ' ' << predictionStatistics.at(threshold)[4] << ' ';
	ss << std::setprecision(defaultPrecision);
	for (int j = 0; j < 2; ++j) {
		rows[currentRow].emplace_back("n/a");
	}
	while (ss >> entry) {
		rows[currentRow].emplace_back(entry);
	}
}

void createClassStatRow(const std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass,
	const std::string& className, double classInstances, const std::string& thresholdName,
	double thresholdValue, std::vector<std::vector<std::string> >& rows, size_t currentRow) {
	std::stringstream ss;
	std::string entry;
	int defaultPrecision = ss.precision();
	rows[currentRow].push_back(className);
	rows[currentRow].push_back(std::to_string(classInstances));
	rows[currentRow].push_back(std::to_string(thresholdValue));
	rows[currentRow].push_back("n/a");
	ss << std::fixed << std::setprecision(2);
	for (double statistic : predictionStatisticsPerClass.at(thresholdName + "|" + className)) {
		ss << statistic << ' ';
	}
	ss << std::setprecision(defaultPrecision);
	while (ss >> entry) {
		rows[currentRow].emplace_back(entry);
	}
}

void createStatisticRowsForThreshold(const std::unordered_map<std::string, double[5]>& predictionStatistics,
	const std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass,
	const std::unordered_map<std::string, double>& numInstancesPerClass,
	const std::vector<std::string>& classNames, const std::string& threshold, std::vector<std::vector<std::string> >& rows,
	size_t& currentRow) {
	createOverallStatRow(predictionStatistics, threshold, rows, currentRow);
	++currentRow;
	for (const std::string& className : classNames) {
		createClassStatRow(predictionStatisticsPerClass, className, numInstancesPerClass.at(className),
			threshold, predictionStatistics.at(threshold)[0], rows, currentRow);
		++currentRow;
	}
}

// Writing the summary to a CSV is currently only implemented for the case with default p values
void writeSummaryToCSV(const std::unordered_map<std::string, double[5]>& predictionStatistics,
	const std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass,
	const std::unordered_map<std::string, double>& numInstancesPerClass,
	std::vector<std::string> classNames, const std::string& filename) {
	std::sort(classNames.begin(), classNames.end());
	std::vector<std::string> header = { "class", "totalInstances", "pValueThreshold", "averageClassesOverThreshold",
		"correct(%)", "incorrect(%)", "noneOfTheAbove(%)" };
	std::vector<std::vector<std::string> > rows((TOTAL_DYNAMIC_PVALUES + CONSTANT_PVALUE_THRESHOLDS.size()) * (classNames.size() + 1));

	// Rows for dynamic p values
	std::string threshold;
	size_t currentRow = 0;
	for (int i = TOTAL_DYNAMIC_PVALUES - 1; i >= 0; --i) {
		threshold = std::string("1/") + std::to_string(std::max(PVALUE_INCREMENT * i, 1.0)) + "n";
		createStatisticRowsForThreshold(predictionStatistics, predictionStatisticsPerClass, numInstancesPerClass,
			classNames, threshold, rows, currentRow);
	}

	// Rows for constant p values
	std::vector<double> constantThresholds = CONSTANT_PVALUE_THRESHOLDS;
	std::sort(constantThresholds.begin(), constantThresholds.end());
	for (double constantThreshold : constantThresholds) {
		createStatisticRowsForThreshold(predictionStatistics, predictionStatisticsPerClass, numInstancesPerClass,
			classNames, std::to_string(constantThreshold), rows, currentRow);
	}

	writeToCSV(header, rows, filename);
}

// Tally the total predicted correctly, incorrectly, and as none of the above, and the number of true/false positives and
// false negatives per class for the given test dataset
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
				if (largestPValue->first == expectedClass) {
					++TEST_RESULTS.overallPredStats[threshold][2];
					++TEST_RESULTS.confusionMatrix[threshold + "|" + largestPValue->first][0];
				}
				else {
					++TEST_RESULTS.overallPredStats[threshold][3];
					++TEST_RESULTS.confusionMatrix[threshold + "|" + largestPValue->first][1];
					++TEST_RESULTS.confusionMatrix[threshold + "|" + expectedClass][2];
				}
			}
			else {
				++TEST_RESULTS.overallPredStats[threshold][4];
			}

			// Count how many classes were over the p value threshold
			for (const auto& pair : pvalues[i])
				if (pair.second >= pvalueThresholds[j]) ++TEST_RESULTS.overallPredStats[threshold][1];
		}
	}
}

void calculateSummary() {
	// Determine size of entire dataset used for training/testing
	size_t total = 0;
	for (const auto& pair : MODEL_STATE.numInstancesPerClass) {
		total += pair.second;
	}

	for (auto& pair : TEST_RESULTS.overallPredStats) {
		for (const std::string& className : MODEL_STATE.classNames) {
			// Calculate number of true negatives for each class using the current p value threshold
			std::string classKey = pair.first + "|" + className;
			double* classConfusionMatrix = TEST_RESULTS.confusionMatrix[classKey];
			classConfusionMatrix[3] = pair.second[2] - classConfusionMatrix[0];

			// Calculate the precision for each class using the current p value threshold
			TEST_RESULTS.precision[classKey] = (classConfusionMatrix[0] / (classConfusionMatrix[0] + classConfusionMatrix[1])) * 100;
		}

		// Calculate accuracy statistics across the entire dataset using the current p value threshold
		pair.second[0] /= K_FOLDS;
		pair.second[1] /= total;
		std::transform(pair.second + 2, pair.second + 5, pair.second + 2,
			[total](double stat) {return (stat / total) * 100; });
	}
}

void printPredCategories(const std::string& threshold) {
	int defaultPrecision = std::cout.precision();
	std::cout << std::setprecision(2) << std::fixed;
	std::cout << "Average number of classes over the p value threshold per datapoint: " <<
		TEST_RESULTS.overallPredStats[threshold][1] << std::endl;
	std::cout << "Correct: " << TEST_RESULTS.overallPredStats[threshold][2] << "%\n";
	std::cout << "Incorrect: " << TEST_RESULTS.overallPredStats[threshold][3] << "%\n";
	std::cout << "None of the above: " << TEST_RESULTS.overallPredStats[threshold][4] << "%\n\n";
	std::cout << std::setprecision(defaultPrecision);
	std::cout.unsetf(std::ios::fixed);
}

void printPredCategoriesPerClass(const std::string& classKey) {
	// Print out confusion matrix for class
	std::cout << "Confusion Matrix:\n";
	std::cout << "|-----|-----|" << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "| " << std::setw(4) << TEST_RESULTS.confusionMatrix.at(classKey)[0] << "| " << std::setw(4) <<
		TEST_RESULTS.confusionMatrix.at(classKey)[1] << "|" << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "|-----|-----|" << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "| " << std::setw(4) << TEST_RESULTS.confusionMatrix.at(classKey)[2] << "| " << std::setw(4) <<
		TEST_RESULTS.confusionMatrix.at(classKey)[3] << "|" << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "|" << std::right << std::setw(6) << "|" << std::setw(6) << "|" << std::left << std::endl;
	std::cout << "|-----|-----|" << std::endl << std::endl;

	// Print out precision for class
	int defaultPrecision = std::cout.precision();
	std::cout << "Precision: " << std::setprecision(2) << std::fixed << TEST_RESULTS.precision[classKey] << "%\n\n";
	std::cout << std::setprecision(defaultPrecision);
	std::cout.unsetf(std::ios::fixed);
}

void printPredCategoriesAllClasses(const std::string& threshold) {
	for (const std::string& className : MODEL_STATE.classNames) {
		std::cout << "Class: " << className << std::endl;
		std::cout << "Total instances: " << MODEL_STATE.numInstancesPerClass.at(className) << std::endl;
		printPredCategoriesPerClass(threshold + "|" + className);
	}
}

// Print out summary of percentages correct, incorrect, and none of the above, and the average number of classes over
// the p value threshold
void printSummary() {
	std::cout << "\n--------------------------" << std::endl;
	std::cout << "\nSummary Statistics (n is the greatest number of datapoints belonging to a class):\n";

	// Print out summary when using user defined p value threshold
	if (TEST_RESULTS.pvalueThreshold >= 0) {
		std::cout << "\n--------------------------\n";
		std::cout << "User defined p value threshold: " << TEST_RESULTS.pvalueThreshold << std::endl;
		printPredCategories(std::to_string(TEST_RESULTS.pvalueThreshold));
		printPredCategoriesAllClasses(std::to_string(TEST_RESULTS.pvalueThreshold));
		return;
	}

	// Print out summary when using default p value thresholds
	for (double constThreshold : CONSTANT_PVALUE_THRESHOLDS) {
		std::cout << "\n--------------------------\n";
		std::cout << "Constant p value threshold: " << constThreshold << std::endl;
		printPredCategories(std::to_string(constThreshold));
		printPredCategoriesAllClasses(std::to_string(constThreshold));
	}

	std::string threshold;
	for (int i = 0; i < TOTAL_DYNAMIC_PVALUES; ++i) {
		threshold = std::string("1/") + std::to_string(std::max(PVALUE_INCREMENT * i, 1.0)) + "n";
		std::cout << "\n--------------------------\n";
		std::cout << "Dynamic p value threshold: " << threshold << " = " << TEST_RESULTS.overallPredStats.at(threshold)[0] << std::endl;
		printPredCategories(threshold);
		printPredCategoriesAllClasses(threshold);
	}	
}

void test(std::vector<ClassMember>& dataset, size_t fold, bool bestFitFunctionsToCSV,
	const std::string& bestFitFunctionsCSVFilename, bool pValuesToCSV, const std::string& pValuesCSVFilename,
	bool summaryToCSV, const std::string& summaryCSVFilename) {
	std::vector<double> pvalueThresholds = createPValueThresholds();

	std::vector<ClassMember> standardizedDataset = standardize(dataset);

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

		threads[i] = std::thread{ calculatePValues, std::cref(nnDistances), std::ref(pvalues), start, stop };

		start = stop;
	}

	for (auto& t : threads) {
		t.join();
	}

	if (bestFitFunctionsToCSV) {
		writeBestFitFunctionsToCSV(bestFitFunctionsCSVFilename, fold);
	}

	printPValues(dataset, pvalues);
	if (pValuesToCSV) {
		writePValuesToCSV(dataset, pvalues, pValuesCSVFilename, fold);
	}

	// Calculate the percentage of datapoints predicted correctly, incorrectly, or as none of the above
	calculateStatistics(dataset, pvalues, pvalueThresholds);

	// Print out statistics if this is the last fold
	if (fold == K_FOLDS - 1) {
		calculateSummary();
		printSummary();
	}
}
