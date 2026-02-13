#ifndef TESTRESULTS_H
#define TESTRESULTS_H

#include <unordered_map>
#include <vector>
#include <string>
#include "test.h"
#include "NOTAPoints.h"

/*
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
	//std::unordered_map<std::string, double> perClassThresholds;
	std::unordered_map<std::string, std::vector<double> > perClassThresholds;

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
*/

struct TestResults {
	std::unordered_map<NOTACategory, double[3]> overallHyperspacePredStats[PVALUE_NUMERATOR_MAX];
	double overallVoidPredStats[PVALUE_NUMERATOR_MAX][NUM_NN_STEPS][3];
	std::unordered_map<NOTACategory, std::unordered_map<std::string, double[4]> > hyperspaceConfusionMatrix[PVALUE_NUMERATOR_MAX];
	std::unordered_map<std::string, double[4]> voidConfusionMatrix[PVALUE_NUMERATOR_MAX][NUM_NN_STEPS];
	std::unordered_map<NOTACategory, std::unordered_map<std::string, double> > hyperspacePrecision[PVALUE_NUMERATOR_MAX];
	std::unordered_map<std::string, double> voidPrecision[PVALUE_NUMERATOR_MAX][NUM_NN_STEPS];
	std::unordered_map<NOTACategory, double[6]> hyperspaceRandomPoints[PVALUE_NUMERATOR_MAX];
	double voidRandomPoints[PVALUE_NUMERATOR_MAX][NUM_NN_STEPS][6];
	double pvalueThreshold = -1;
	std::unordered_map<std::string, double[PVALUE_NUMERATOR_MAX]> perClassThresholds;

	TestResults() {
		for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
			for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
				for (size_t j = 0; j < 3; ++j) {
					overallVoidPredStats[pvalCat][i][j] = 0.0;
				}
			}

			for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
				for (size_t j = 0; j < 6; ++j) {
					voidRandomPoints[pvalCat][i][j] = 0.0;
				}
			}
		}
	}
};

extern TestResults TEST_RESULTS;

#endif
