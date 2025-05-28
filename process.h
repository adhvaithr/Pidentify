#ifndef PROCESS_H
#define PROCESS_H

#include "classMember.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <ap.h>

std::unordered_map<std::string, std::vector<double> > process(std::unordered_map<std::string, std::vector<ClassMember> >& dataset);
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);
double weightedEuclideanDistance(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& weights);
void removeFeatures(const std::vector<size_t>& indices, std::vector<ClassMember>& dataset);
void projectOntoPrincipalAxes(const alglib::real_2d_array& datapoints, const alglib::real_2d_array& principalAxes,
    alglib::real_2d_array& principalComponents);
//void copyDatapoints(std::unordered_map<std::string, std::vector<ClassMember> >& dataset, alglib::real_2d_array& datapoints, bool to_alglib_array);

#endif