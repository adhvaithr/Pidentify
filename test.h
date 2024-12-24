#ifndef TEST_H
#define TEST_H

#include <vector>
#include "classMember.h"

void trainTestSplit(std::vector<ClassMember>& dataset, std::vector<ClassMember>& testDataset, double testSize);
void test(const std::vector<ClassMember>& dataset, double pvalueThreshold);

#endif
