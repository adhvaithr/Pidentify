#ifndef TESTRESULTS_H
#define TESTRESULTS_H

#include <unordered_map>
#include "test.h"

struct TestResults {
	//std::unordered_map<std::string, double[5]> overallPredStats;
	//std::unordered_map<std::string, double[6]> overallPredStats;
	double overallPredStats[3] = { 0 };
	std::unordered_map<std::string, double[4]> confusionMatrix;
	std::unordered_map<std::string, double> precision;
	double pvalueThreshold = -1;
	std::unordered_map<std::string, double> perClassThresholds;

	std::unordered_map<std::string, size_t> NOTACount;
	std::unordered_map<std::string, size_t> ambiguousCount;

	double randomPoints[6] = { 0 };
};

extern TestResults TEST_RESULTS;

#endif
