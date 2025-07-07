#ifndef TESTRESULTS_H
#define TESTRESULTS_H

#include <unordered_map>
#include "test.h"

struct TestResults {
	std::unordered_map<std::string, double[5]> overallPredStats;
	std::unordered_map<std::string, double[4]> confusionMatrix;
	std::unordered_map<std::string, double> precision;
	double pvalueThreshold;
};

extern TestResults TEST_RESULTS;

#endif
