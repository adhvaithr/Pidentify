#ifndef TEST_H
#define TEST_H

#include <vector>
#include <unordered_map>
#include <string>
#include "classMember.h"

const double PVALUE_INCREMENT = 5.0;
const int TOTAL_DYNAMIC_PVALUES = 5;
const std::vector<double> CONSTANT_PVALUE_THRESHOLDS = { 0.50, 0.30, 0.10, 0.05 };

void kFoldSplit(std::vector<ClassMember>& dataset, std::vector<ClassMember> kSets[]);
void test(const std::vector<ClassMember>& dataset, std::unordered_map<std::string, double[5]>& predictionStatistics, std::unordered_map<std::string, double[3]>& predictionStatisticsPerClass,
	std::unordered_map<std::string, double>& numInstancesPerClass,
	size_t fold, bool applyPCA, double pvalueThreshold, bool bestFitFunctionsToCSV,
	const std::string& bestFitFunctionsCSVFilename, bool pValuesToCSV, const std::string& pValuesCSVFilename,
	bool summaryToCSV, const std::string& summaryCSVFilename);
#endif
