#ifndef TEST_H
#define TEST_H

#include <vector>
#include <unordered_map>
#include <string>
#include <limits>
#include "classMember.h"

const double PVALUE_INCREMENT = 5.0;
const int TOTAL_DYNAMIC_PVALUES = 5;
const std::vector<double> CONSTANT_PVALUE_THRESHOLDS = { 0.50, 0.30, 0.10, 0.05 };

void kFoldSplit(std::vector<ClassMember>& dataset, std::vector<ClassMember> kSets[]);
void test(const std::vector<ClassMember>& dataset, std::unordered_map<std::string, double[5]>& predictionStatistics, size_t fold, double pvalueThreshold = std::numeric_limits<double>::lowest());

#endif
