#include <iostream>
#include <numeric>
#include <thread>

#include "modelState.h"

void ModelState::setDatasetSize() {
	datasetSize = std::accumulate(numInstancesPerClass.begin(), numInstancesPerClass.end(), 0,
		[](size_t a, const std::pair<std::string, double>& b) {return a + b.second; });
}

void ModelState::setWeightExp(const std::string& weightScheme) {
	if (weightScheme == "squared") {
		weightExp = 2;
	}
	else if (weightScheme == "linear") {
		weightExp = 1;
	}
	else if (weightScheme == "unweighted") {
		weightExp = 0;
	}
	else if (weightScheme == "cube root") {
		weightExp = 1.0 / 3.0;
	}
	else {
		std::cerr << "ERROR: Weight scheme must be one of: \"squared\", \"linear\", \"unweighted\", \"cube root\"\n";
		std::exit(0);
	}
}

void ModelState::clearTemporaries() {
	zeroStdDeviation.clear();
	classMap.clear();
	if (!preexistingBestfit) {
		bestFit.clear();
	}
}

// Initialize number of threads to use concurrently
void setThreads() {
	NUM_THREADS = std::max(static_cast<int>(std::thread::hardware_concurrency()), 4);
}
