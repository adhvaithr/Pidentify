#ifndef TESTRESULTS_H
#define TESTRESULTS_H

#include <unordered_map>
#include "test.h"

struct TestResults {
	std::unordered_map<NOTACategory, double[3]> overallHyperspacePredStats;
	double overallVoidPredStats[NUM_NN_STEPS][3];
	std::unordered_map<NOTACategory, std::unordered_map<std::string, double[4]> > hyperspaceConfusionMatrix;
	std::unordered_map<std::string, double[4]> voidConfusionMatrix[NUM_NN_STEPS];
	std::unordered_map<NOTACategory, std::unordered_map<std::string, double> > hyperspacePrecision;
	std::unordered_map<std::string, double> voidPrecision[NUM_NN_STEPS];
	std::unordered_map<NOTACategory, double[6]> hyperspaceRandomPoints;
	double voidRandomPoints[NUM_NN_STEPS][6];
	double pvalueThreshold = -1;

	TestResults() {
		for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
			for (size_t j = 0; j < 3; ++j) {
				overallVoidPredStats[i][j] = 0.0;
			}
			for (size_t j = 0; j < 6; ++j) {
				voidRandomPoints[i][j] = 0.0;
			}
		}
	}
};

extern TestResults TEST_RESULTS;

#endif
