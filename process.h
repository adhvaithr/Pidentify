#ifndef PROCESS_H
#define PROCESS_H

#include "classMember.h"
#include <vector>
#include <unordered_map>
#include <string>

std::unordered_map<std::string, std::vector<double> > process(std::vector<ClassMember> dataset);
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);

#endif