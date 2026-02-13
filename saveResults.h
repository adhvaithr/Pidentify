#ifndef SAVERESULTS_H
#define SAVERESULTS_H

#include <string>
#include <vector>
#include <unordered_map>

#include "classMember.h"

void writeBestFitFunctionsToCSV(const std::string& filename, int fold = -1);
void writePValuesToCSV(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& pvalues, size_t fold);
void writeNOTACategoryResultsToCSV();
void cacheTestPlotInfo(const std::vector<std::pair<std::string, std::string> >& classifications,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold);

#endif