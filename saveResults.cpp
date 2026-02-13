#include <cstdio>

#include "saveResults.h"
#include "cachePaths.h"
#include "modelState.h"
#include "testResults.h"
#include "NOTAPoints.h"

void writeBestFitFunctionsToCSV(const std::string& filename, int fold) {
	FILE* fp;

	if (fold <= 0) {
		std::string header;
		if (fold == 0) {
			header += "fold,";
		}
		header += "class,bestFitFunction,c,a,residual";

		fp = fopen(filename.c_str(), "w");
		fprintf(fp, "%s\n", header.c_str());
	}
	else {
		fp = fopen(filename.c_str(), "a");
	}

	for (const auto& pair : MODEL_STATE.bestFit) {
		if (fold >= 0) {
			fprintf(fp, "%d,", fold);
		}

		fprintf(fp, "%s,%s,%g,%g,%g\n", pair.first.c_str(), pair.second.functionName.c_str(),
			pair.second.c[0], pair.second.c[1], pair.second.wrmsError);
	}

	fclose(fp);
}

void writePValuesToCSV(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues, size_t fold) {
	FILE* fp;
	char delim;
	size_t totalDatapoints = pvalues.size();
	size_t totalClasses = MODEL_STATE.classNames.size();

	// Create a header if this is the first time writing to the CSV file
	if (fold == 0) {
		fp = fopen(CACHE_PATHS.pvaluesFilepath.c_str(), "w");
		fprintf(fp, "lineNumber,fold,");
		for (size_t i = 0; i < totalClasses; ++i) {
			delim = (i == totalClasses - 1) ? '\n' : ',';
			fprintf(fp, "%s%c", MODEL_STATE.classNames[i].c_str(), delim);
		}
	}
	else {
		fp = fopen(CACHE_PATHS.pvaluesFilepath.c_str(), "a");
	}

	for (size_t i = 0; i < totalDatapoints; ++i) {
		fprintf(fp, "%lu,%lu,", dataset[i].lineNumber, fold);
		for (size_t j = 0; j < totalClasses; ++j) {
			delim = (j == totalClasses - 1) ? '\n' : ',';
			std::string className = MODEL_STATE.classNames[j];
			if (pvalues[i].find(className) != pvalues[i].end()) {
				fprintf(fp, "%g%c", pvalues[i].at(className), delim);
			}
			else {
				fprintf(fp, "0%c", delim);
			}
		}
	}

	fclose(fp);
}

/*
void writeNOTACategoryResultsToCSV() {
	FILE* fp = fopen(CACHE_PATHS.NOTACategoryResultsFilepath.c_str(), "w");

	fprintf(fp, "category,minNNUnitsFromClass,totalPoints,");
	for (const std::string& className : MODEL_STATE.classNames) {
		fprintf(fp, "%sPrecision,", className.c_str());
	}
	fprintf(fp, "recall\n");

	for (int i = 0; i < NUM_NN_STEPS; ++i) {
		fprintf(fp, "%i,%g,%g,", NOTACategory::VOID, NNStepMultiplier(i), TEST_RESULTS.voidRandomPoints[i][0]);
		for (const std::string& className : MODEL_STATE.classNames) {
			fprintf(fp, "%.2f,", TEST_RESULTS.voidPrecision[i][className]);
		}
		fprintf(fp, "%.2f\n", TEST_RESULTS.voidRandomPoints[i][5]);
	}

	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		NOTACategory NOTALoc = static_cast<NOTACategory>(i);
		fprintf(fp, "%lu,-1,%g,", i, TEST_RESULTS.hyperspaceRandomPoints[NOTALoc][0]);
		for (const std::string& className : MODEL_STATE.classNames) {
			fprintf(fp, "%.2f,", TEST_RESULTS.hyperspacePrecision[NOTALoc][className]);
		}
		fprintf(fp, "%.2f\n", TEST_RESULTS.hyperspaceRandomPoints[NOTALoc][5]);
	}

	fclose(fp);
}
*/

void writeNOTACategoryResultsToCSV() {
	FILE* fp = fopen(CACHE_PATHS.NOTACategoryResultsFilepath.c_str(), "w");

	fprintf(fp, "category,minNNUnitsFromClass,totalPoints,pvalueNumerator,");
	for (const std::string& className : MODEL_STATE.classNames) {
		fprintf(fp, "%sPrecision,", className.c_str());
	}
	fprintf(fp, "recall\n");

	for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
		for (int i = 0; i < NUM_NN_STEPS; ++i) {
			fprintf(fp, "%i,%g,%g,%i,", NOTACategory::VOID, NNStepMultiplier(i), TEST_RESULTS.voidRandomPoints[pvalCat][i][0], pvalCat + 1);
			for (const std::string& className : MODEL_STATE.classNames) {
				fprintf(fp, "%.2f,", TEST_RESULTS.voidPrecision[pvalCat][i][className]);
			}
			fprintf(fp, "%.2f\n", TEST_RESULTS.voidRandomPoints[pvalCat][i][5]);
		}

		for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
			NOTACategory NOTALoc = static_cast<NOTACategory>(i);
			fprintf(fp, "%lu,-1,%g,%i,", i, TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTALoc][0], pvalCat + 1);
			for (const std::string& className : MODEL_STATE.classNames) {
				fprintf(fp, "%.2f,", TEST_RESULTS.hyperspacePrecision[pvalCat][NOTALoc][className]);
			}
			fprintf(fp, "%.2f\n", TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTALoc][5]);
		}
	}

	fclose(fp);
}

void cacheTestPlotInfo(const std::vector<std::pair<std::string, std::string> >& classifications,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold) {
	if (fold == 0) {
		createFolder(CACHE_PATHS.classificationsDirectory.c_str());
	}
	std::string filepath = CACHE_PATHS.classificationsDirectory + getPathSep() + "classifications-iter" +
		std::to_string(fold) + ".csv";
	FILE* fp = fopen(filepath.c_str(), "w");
	size_t testSize = nnDistances.size();

	fprintf(fp, "className,classification");
	for (auto iter = nnDistances[0].begin(); iter != nnDistances[0].end(); ++iter) {
		fprintf(fp, ",%s", iter->first.c_str());
	}
	fprintf(fp, "\n");

	for (size_t i = 0; i < testSize; ++i) {
		fprintf(fp, "%s,%s", classifications[i].first.c_str(), classifications[i].second.c_str());
		for (auto iter = nnDistances[i].begin(); iter != nnDistances[i].end(); ++iter) {
			fprintf(fp, ",%g", iter->second);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}