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

std::vector<ClassMember> normalize(std::vector<ClassMember> dataset) {
	size_t numFeatures = dataset[0].features.size();
	for (auto& obj : dataset) {
		if (obj.features.size() != numFeatures) {
			std::cerr << "Inconsistent feature size: " << numFeatures << " != " << obj.features.size() << std::endl;
			std::exit(0);
		}
		for (size_t i = 0; i < numFeatures; ++i) {
			obj.features[i] = (obj.features[i] - MODEL_STATE.featureMeans[i]) / MODEL_STATE.minRadius;
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
 
// Calculate the percentage predicted correctly, incorrectly, and as none of the above, and the average number of classes per
// datapoint that are over the p value threshold
void calculateStatistics(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::vector<double>& pvalueThresholds, std::unordered_map<std::string, double[5]>& predictionStatistics, 
    std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass, std::unordered_map<std::string, double>& numInstancesPerClass,
    bool userPValueThreshold) {

	size_t total = dataset.size(), numThresholds = pvalueThresholds.size();
	std::vector<double> numCorrect(numThresholds, 0), numIncorrect(numThresholds, 0), numNOTA(numThresholds, 0),
		classesOverThreshold(numThresholds, 0);

    std::unordered_map<std::string, std::vector<double>> numCorrectPerClass;
    std::unordered_map<std::string, std::vector<double>> numIncorrectPerClass;
    std::unordered_map<std::string, std::vector<double>> numNOTAPerClass;
    std::set<std::string> seenClasses = {};
    std::unordered_map<std::string, double> newInstancesPerClass;
	for (size_t i = 0; i < total; ++i) {
		// Find the largest p value for the datapoint
        const std::string& expectedClass = dataset[i].name;
        if (seenClasses.find(expectedClass) == seenClasses.end()) {
            numCorrectPerClass[expectedClass] = std::vector<double>(numThresholds, 0);
            numIncorrectPerClass[expectedClass] = std::vector<double>(numThresholds, 0);
            numNOTAPerClass[expectedClass] = std::vector<double>(numThresholds, 0);
            seenClasses.insert(expectedClass);
        }
       
        numInstancesPerClass[expectedClass]+=1;
        newInstancesPerClass[expectedClass]+=1;
        
		auto largestPValue = std::max_element(pvalues[i].begin(), pvalues[i].end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2) {
			return p1.second < p2.second;
			});

		for (size_t j = 0; j < numThresholds; ++j) {
			// Tally whether the p value indicates the correct/incorrect class or none of the above (NOTA)
			if (largestPValue->second >= pvalueThresholds[j]) {
              
                if (largestPValue->first == expectedClass) {
                    ++numCorrect[j];
                    ++numCorrectPerClass[expectedClass][j];
                }
                else {
                    ++numIncorrect[j];
                    ++numIncorrectPerClass[expectedClass][j];
                }
            }
            else {
                ++numNOTA[j];
                ++numNOTAPerClass[expectedClass][j];
            }

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

        /* Add class specific percentages for correct, incorrect, and NOTA predictions*/
        for (const auto& currClass : newInstancesPerClass) {
            const std::string& className = currClass.first;
            double classCount = currClass.second;
 
            std::string classKey = threshold + "|" + className;
            // predictionStatisticsPerClass[classKey][0] = classKey;
            predictionStatisticsPerClass[classKey][0] += (numCorrectPerClass[className][i] / classCount) * 100;
            predictionStatisticsPerClass[classKey][1] += (numIncorrectPerClass[className][i] / classCount) * 100;
            predictionStatisticsPerClass[classKey][2] += (numNOTAPerClass[className][i] / classCount) * 100;
 
 
        }
	}
}

// Calculate the average p value, classes over the p value threshold, correct, incorrect, and none of the above across k folds
void calculateSummary(std::unordered_map<std::string, double[5]>& predictionStatistics, std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass, int kFolds) {
	for (auto& pair : predictionStatistics) {
		for (double& val : pair.second) {
			val /= kFolds;
		}
	}

    for (auto& pair : predictionStatisticsPerClass) {
		for (double& val : pair.second) {
			val /= kFolds;
		}
	}
}


void printPredCategories(const double results[]) {
	int defaultPrecision = std::cout.precision();
	std::cout << "Average number of classes over the p value threshold per datapoint: " << results[1] << std::endl;
    std::cout << "==========================" << std::endl;
	std::cout << std::setprecision(2) << std::fixed;
	std::cout << "Correct: " << results[2] << "%\n";
	std::cout << "Incorrect: " << results[3] << "%\n";
	std::cout << "None of the above: " << results[4] << "%\n";
	std::cout << std::setprecision(defaultPrecision);
	std::cout.unsetf(std::ios::fixed);
    std::cout << "==========================" << std::endl;
}

void printPredCategoriesPerClass(const double results[]) {
    int defaultPrecision = std::cout.precision();
    std::cout << "--------------------------" << std::endl;
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Correct: " << results[0] << "%\n";
    std::cout << "Incorrect: " << results[1] << "%\n";
    std::cout << "None of the above: " << results[2] << "%\n";
    std::cout << std::setprecision(defaultPrecision);
    std::cout.unsetf(std::ios::fixed);
    std::cout << "--------------------------" << std::endl;
 }

void printPredCategoriesAllClasses(const std::string& threshold,
	const std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass,
	const std::vector<std::string>& classNames) {
	for (auto& currClass : classNames) {
		std::string classKey = threshold + "|" + currClass;
		std::cout << "Class: " << currClass << std::endl;
		printPredCategoriesPerClass(predictionStatisticsPerClass.at(classKey));
	}
}
 
void printSummaryPerClass(const std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass) {

    for (auto& classStats : predictionStatisticsPerClass) {
        std::cout << "\nSummary Statistics: for " << classStats.first << " \n";
        printPredCategoriesPerClass(classStats.second);
    }
 }
 
// Print out summary of percentages correct, incorrect, and none of the above, and the average number of classes over
// the p value threshold when using p value threshold specified by the user
void printSummary(const double statistics[]) {
    std::cout << "--------------------------" << std::endl;
	std::cout << "\nSummary Statistics:\n";
	printPredCategories(statistics);
}


// Print out summary of percentages correct, incorrect, and none of the above, and the average number of classes over
// the p value threshold when using default p value thresholds
void printSummary(const std::unordered_map<std::string, double[5]>& predictionStatistics,
   const std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass,
   std::vector<std::string>& classNames) {
    std::cout << "--------------------------" << std::endl;
	std::cout << "\nSummary Statistics (n is the greatest number of datapoints belonging to a class):\n";

	std::string threshold;
	for (int i = TOTAL_DYNAMIC_PVALUES - 1; i >= 0; --i) {
		threshold = std::string("1/") + std::to_string(std::max(PVALUE_INCREMENT * i, 1.0)) + "n";
		std::cout << "\nDynamic p value threshold: " << threshold << " = " << predictionStatistics.at(threshold)[0] << std::endl;
		printPredCategories(predictionStatistics.at(threshold));
		printPredCategoriesAllClasses(threshold, predictionStatisticsPerClass, classNames);
	}

    
	std::vector<double> constantThresholds = CONSTANT_PVALUE_THRESHOLDS;
	std::sort(constantThresholds.begin(), constantThresholds.end());
	for (double constThreshold : constantThresholds) {
		std::cout << "Constant p value threshold: " << constThreshold << std::endl;
		printPredCategories(predictionStatistics.at(std::to_string(constThreshold)));
		printPredCategoriesAllClasses(std::to_string(constThreshold), predictionStatisticsPerClass, classNames);
	}
}

void printClassCounts(const std::unordered_map<std::string, double>& classCounts) {
    std::cout << "--------------------------" << std::endl;
    std::cout << "Class Counts (Label: Number of Instances)" << std::endl;
    for (const auto& item : classCounts) {
        std::cout << item.first << ": " << item.second << std::endl;
    }
}

void test(const std::vector<ClassMember>& dataset, std::unordered_map<std::string, double[5]>& predictionStatistics, std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass,
    std::unordered_map<std::string, double>& numInstancesPerClass,
	size_t fold, bool applyPCA, double pvalueThreshold, bool bestFitFunctionsToCSV,
	const std::string& bestFitFunctionsCSVFilename, bool pValuesToCSV, const std::string& pValuesCSVFilename,
	bool summaryToCSV, const std::string& summaryCSVFilename) {
	// Create p value thresholds if none are provided by the user
	std::vector<double> pvalueThresholds;
	bool userPValueThreshold;
	if (pvalueThreshold == -1) {
		pvalueThresholds = createPValueThresholds(predictionStatistics);
		userPValueThreshold = false;
	}
	else {
		pvalueThresholds.push_back(pvalueThreshold);
		predictionStatistics[std::to_string(pvalueThreshold)][0] += pvalueThreshold;
		userPValueThreshold = true;
	}

	std::vector<ClassMember> normalizedDataset = normalize(dataset);

	
	if (applyPCA && dataset[0].features.size() >= 3) {
		toPCASubspace(normalizedDataset);
	}

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
	
	if (bestFitFunctionsToCSV) {
		writeBestFitFunctionsToCSV(bestFitFunctionsCSVFilename, fold);
	}

	printPValues(dataset, pvalues);
	if (pValuesToCSV) {
		writePValuesToCSV(dataset, pvalues, pValuesCSVFilename, fold);
	}

	// Calculate the percentage of datapoints predicted correctly, incorrectly, or as none of the above
	calculateStatistics(dataset, pvalues, pvalueThresholds, 
        predictionStatistics, predictionStatisticsPerClass, numInstancesPerClass, userPValueThreshold);

    std::vector<std::string> classNames;
    for (const auto& classItem : numInstancesPerClass) {
        classNames.push_back(classItem.first);
    }
     
	// Print out statistics if this is the last fold
	if (fold == K_FOLDS - 1) {
		calculateSummary(predictionStatistics, predictionStatisticsPerClass, K_FOLDS);
        printClassCounts(numInstancesPerClass);
		if (userPValueThreshold) {
			printSummary(predictionStatistics[std::to_string(pvalueThreshold)]);
            printSummaryPerClass(predictionStatisticsPerClass);
		}
		else {
            printSummary(predictionStatistics, predictionStatisticsPerClass, classNames);
			if (summaryToCSV) {
				writeSummaryToCSV(predictionStatistics, predictionStatisticsPerClass, numInstancesPerClass, classNames,
					summaryCSVFilename);
			}
		}
	}
}