#ifndef TEST_H
#define TEST_H

#include <vector>
#include <unordered_map>
#include <string>
#include "classMember.h"

void test(std::vector<ClassMember>& dataset, size_t fold);
void kFoldSplit(std::unordered_map<std::string, std::vector<ClassMember> >& dataset,
	std::unordered_map<std::string, std::vector<ClassMember> > kSets[], size_t maxPerClass = 1000);
void findFeatureBB(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset, double extension = 2);
void setPValueThreshold(const std::string& threshold);
void insertRandomPoints(std::vector<ClassMember>& dataset, size_t numInsert, size_t iteration);

void test(const std::vector<ClassMember>& dataset,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold);
void test(const std::vector<ClassMember>& dataset, const std::vector<std::unordered_map<std::string, double> >& pvalues,
	const std::vector<std::unordered_map<std::string, double> >& nnDistances, size_t fold);

#endif
