#include <sys/stat.h>

#include "cachePaths.h"

void CachePaths::initPaths(const std::string& dir) {
	std::string pathSep = getPathSep();
	cacheDirectory = (dir == "" || dir == "." || dir == "." + pathSep) ? "." : dir;
	bestFitFunctionsFilepath = cacheDirectory + pathSep + "bestFitFunctions.csv";
	pvaluesFilepath = cacheDirectory + pathSep + "pvalues.csv";
	ecdfDirectory = cacheDirectory + pathSep + "ecdf_info";
	classificationsDirectory = cacheDirectory + pathSep + "classifications_info";
	NOTAPointsFilepath = cacheDirectory + pathSep + "NOTAPoints.csv";
	NOTACategoryResultsFilepath = cacheDirectory + pathSep + "resultsByNOTACategory.csv";
	finishedFilepath = cacheDirectory + pathSep + "isFinished.txt";
}

std::string getPathSep() {
	#ifdef _WIN32
		return "\\";
	#else
		return "/";
	#endif
}

void createFolder(const char* dir) {
	struct stat sb;
	if (stat(dir, &sb) != 0) {
		#ifdef _WIN32
			mkdir(dir);
		#else
			mkdir(dir, 0777);
		#endif
	}
}
