#ifndef MODELSTATE_H
#define MODELSTATE_H

#include <vector>
#include <array>
#include <unordered_map>
#include <string>
#include <mutex>
#include <ap.h>
#include "classMember.h"
#include "fit.h"

struct ModelState {
	std::vector<double> means;
	std::vector<double> sigmas;
	std::vector<size_t> zeroStdDeviation;
	alglib::real_2d_array principalAxes;
	std::unordered_map<std::string, std::vector<std::vector<double> > > classMap;
	bool preexistingBestfit;
	std::unordered_map<std::string, FitResult> bestFit;
	size_t trainDatasetSize;
	size_t datasetSize;
	std::vector<std::string> classNames;
	std::unordered_map<std::string, std::vector<double> > featureWeights;
	std::string processType;
	std::unordered_map<std::string, double> numInstancesPerClass;
	std::vector<double> featureMins;
	std::vector<double> featureMaxs;
	double weightExp;

	void setDatasetSize();
	void setWeightExp(const std::string& weightScheme);
	void clearTemporaries();
};

void setThreads();

extern ModelState MODEL_STATE;
extern std::mutex m;
extern double NUM_THREADS;
extern int K_FOLDS;
extern size_t MIN_CLASS_MEMBERS;
extern size_t MAX_CLASS_MEMBERS;

#endif