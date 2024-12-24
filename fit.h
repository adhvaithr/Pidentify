#ifndef FIT_H
#define FIT_H

#include <vector>
#include <unordered_map>
#include <string>
#include "interpolation.h"

struct FitResult {
    alglib::real_1d_array c;
    std::string functionName;
    double wrmsError;
};

int fitClasses(const std::unordered_map<std::string, std::vector<double> >& sorted_distances);
double logistic(double k, double alpha, double x);
double hyperbolic_tangent(double k, double alpha, double x);
double arctangent(double k, double alpha, double x);
double gudermannian(double k, double alpha, double x);
double algebraic(double k, double alpha, double x);

#endif