#ifndef TEST_H
#define TEST_H

#include <vector>
#include "classMember.h"

void kFoldSplit(std::vector<ClassMember>& dataset, std::vector<ClassMember> kSets[]);
void test(const std::vector<ClassMember>& dataset, double pvalueThreshold, size_t fold);

#endif
