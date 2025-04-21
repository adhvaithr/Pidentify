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
	std::vector<double> featureMeans;
	alglib::real_2d_array principalAxes;
	double minRadius;
	std::unordered_map<std::string, std::vector<ClassMember> > classMap;
	std::unordered_map<std::string, FitResult> bestFit;
};

extern ModelState MODEL_STATE;

extern std::mutex m;

extern double NUM_THREADS;

extern int K_FOLDS;

#endif