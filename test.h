#ifndef TEST_H
#define TEST_H

#include <vector>
#include <unordered_map>
#include <string>
#include "classMember.h"

const static double HYPERSPACE_MAX_BBOX_EXTENSION = 2.0;
const static size_t HYPERSPACE_LOWER_BOUNDS = 11;
const static double HYPERSPACE_BBOX_LOWER_BOUNDS[HYPERSPACE_LOWER_BOUNDS] = { 0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00 };
const static double NN_STEPS_START = 0.1;
const static double NN_STEP_SIZE = 0.1;
const static size_t NUM_NN_STEPS = 15;
const static size_t MIN_NOTA_POINTS = 1000;

void test(std::vector<ClassMember>& dataset, size_t fold);
void kFoldSplit(std::unordered_map<std::string, std::vector<ClassMember> >& dataset,
	std::unordered_map<std::string, std::vector<ClassMember> > kSets[], size_t maxPerClass = 1000);
void findFeatureBB(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset, double extension);
void setPValueThreshold(const std::string& threshold);
std::vector<ClassMember> createNOTAPoints(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset);

void test(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold);
void test(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold);

#endif
