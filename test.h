#ifndef TEST_H
#define TEST_H

#include <vector>
#include <unordered_map>
#include <string>
#include "classMember.h"

const double PVALUE_INCREMENT = 5.0;
const int TOTAL_DYNAMIC_PVALUES = 5;
const std::vector<double> CONSTANT_PVALUE_THRESHOLDS = { 0.50, 0.30, 0.10, 0.05 };

void test(std::vector<ClassMember>& dataset, size_t fold, bool bestFitFunctionsToCSV,
	const std::string& bestFitFunctionsCSVFilename, bool pValuesToCSV, const std::string& pValuesCSVFilename,
	bool summaryToCSV, const std::string& summaryCSVFilename);

#endif
