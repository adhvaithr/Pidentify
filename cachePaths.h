#ifndef CACHEPATHS_H
#define CACHEPATHS_H

#include <string>

struct CachePaths {
	std::string cacheDirectory;
	std::string bestFitFunctionsFilepath;
	std::string pvaluesFilepath;
	std::string ecdfDirectory;
	std::string classificationsDirectory;

	void initPaths(const std::string& dir);
};

std::string getPathSep();
void createFolder(const char*);

extern CachePaths CACHE_PATHS;

#endif