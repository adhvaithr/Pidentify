#ifndef CACHEPATHS_H
#define CACHEPATHS_H

#include <string>

struct CachePaths {
	std::string cacheDirectory;
	std::string bestFitFunctionsFilepath;
	std::string bestFitFunctionsByFoldFilepath;
	std::string pvaluesFilepath;
	std::string ecdfDirectory;
	std::string classificationsDirectory;
	std::string NOTAPointsFilepath;
	std::string NOTACategoryResultsFilepath;
	std::string finishedFilepath;

	void initPaths(const std::string& dir);
};

std::string getPathSep();
void createFolder(const char*);
bool fileExists(const std::string& fileName);

extern CachePaths CACHE_PATHS;

#endif