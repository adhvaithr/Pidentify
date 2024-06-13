#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <limits>
#include <assert.h>
#include <unordered_map>
#include <algorithm>

#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "interpolation.h"

using namespace std;
using namespace alglib;

double sech(double x) {
    return 1.0 / std::cosh(x);
}

// a.Logistic function
double logistic(double k, double alpha, double x){
     return 1.0 / (1.0 + exp(-k*(x-alpha)));
}

void sigf_a(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    // This calculates 1/(1 + e^-(c * (x - a))
    func = 1 - logistic(c[0],c[1],x[0]);
}
void sigf_ad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) {
    func = 1 - logistic(c[0],c[1],x[0]);
    grad[0] = - (((x[0]-c[1]) * exp(c[0] * (c[1] - x[0]))) / (exp(c[0] * (c[1] - x[0])) + 1) * (exp(c[0] * (c[1] - x[0])) + 1));
    grad[1] = c[0] * exp(c[0] * (c[1] - x[0])) / (exp(c[0] * (c[1] - x[0])) + 1) * (exp(c[0] * (c[1] - x[0])));
}

// b.Hyperbolic tangent
double Hyperbolic_tangent(double k, double alpha, double x)
{
    return ((exp(k * (x - alpha)) - exp(-k * (x - alpha)))/(exp(k * (x - alpha)) + exp(-k * (x - alpha))) + 1) / 2;
}

void sigf_b(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - Hyperbolic_tangent(c[0], c[1], x[0]);
}

void sigf_bd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - Hyperbolic_tangent(c[0], c[1], x[0]);
    grad[0] = - (2 * (x[0] - c[1]) * exp(2 * c[0] * (x[0] - c[1]))) / ((exp(2 * c[0] * (x[0] - c[1])) + 1) *  (exp(2 * c[0] * (x[0] - c[1])) + 1));
    grad[1] = (2 * c[0] * exp(2 * c[0] * (x[0] - c[1]))) / ((exp(2 * c[0] * (x[0] - c[1])) + 1) * (exp(2 * c[0] * (x[0] - c[1])) + 1));
}

// c.Arctangent function
double arctangent(double k, double alpha, double x)
{
    return (atan(k * (x - alpha)) + 1) / 2;
}

void sigf_c(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - arctangent(c[0], c[1], x[0]);
}

void sigf_cd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - arctangent(c[0], c[1], x[0]);
    grad[0] = - ((x[0] - c[1]) / (2 * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)));
    grad[1] =  - (c[0] / (2 * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)));
}

// d.Gudermannian function
double gudermannian(double k, double alpha, double x)
{
    return ((2 * atan(tanh(k * (x - alpha)/ 2))) + 1) / 2;
}

void sigf_d(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - gudermannian(c[0], c[1], x[0]);
}

void sigf_dd(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - gudermannian(c[0], c[1], x[0]);
    grad[0] = -((x[0] - c[1]) * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1])))/ (2 * ((tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1)));
    grad[1] = c[0] * sech(1/2 * c[0] * (x[0] - c[1])) * sech(1/2 * c[0] * (x[0] - c[1])) / (2 * (tanh(1/2 * c[0] * (x[0] - c[1])) * tanh(1/2 * c[0] * (x[0] - c[1])) + 1));
}

// e. A simple algebraic function
double algebraic(double k, double alpha, double x)
{
    double term = k * (x - alpha);
    return ((k * (x - alpha)) / (sqrt(1 + term * term)) + 1) / 2;
}

void sigf_e(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr)
{
    func = 1 - algebraic(c[0], c[1], x[0]);
}

void sigf_ed(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
{
    func = 1 - algebraic(c[0], c[1], x[0]);
    grad[0] = - ((x[0] - c[1]) / (2 * (sqrt((c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)))));
    grad[1] = - (c[0] / (2 * (sqrt((c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1) * (c[0] * c[0] * (x[0] - c[1]) * (x[0] - c[1]) + 1)))));
}

// Struct to hold Iris data 
struct Iris {
    std::vector<double> features;
    std::string species;
};

// Read the dataset from a file
std::vector<Iris> readDataset(const std::string& filename) {
    std::vector<Iris> dataset;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;  // Skip the empty lines
        }
        std::stringstream ss(line);
        Iris iris;
        std::string feature;

        while (std::getline(ss, feature, ',')) {
            if (isdigit(feature[0]) || feature[0] == '-') {
                iris.features.push_back(std::stod(feature));
            } else {
                assert(iris.species == "");
                assert(feature != "");
                iris.species = feature;
            }
        }
        dataset.push_back(iris);
    }

    std::cout << "feature size:" << dataset[0].features.size() << std::endl;
    std::cout << dataset[0].features[0] << std::endl;
    return dataset;
}

// Normalize features
void normalizeFeatures(std::vector<Iris>& dataset) {
    if (dataset.empty()) {
        std::cerr << "Dataset is empty!" << std::endl;
        return;
    }

    size_t numFeatures = dataset[0].features.size();
    std::vector<double> means(numFeatures, 0.0);
    std::vector<double> sigmas(numFeatures, 0.0);

    // Calculate mean for each feature
    for (const auto& iris : dataset) {
        if (iris.features.size() != numFeatures) {
            fprintf(stderr, "Inconsistent feature size: %zu != %zu\n", iris.features.size(), numFeatures);
            
            return;
        }
        for (size_t i = 0; i < numFeatures; ++i) {
            means[i] += iris.features[i];
        }
    }

    for (double& mean : means) {
        mean /= dataset.size();
    }

    // Calculate standard deviation for each feature
    for (const auto& iris : dataset) {
        for (size_t i = 0; i < numFeatures; ++i) {
            sigmas[i] += (iris.features[i] - means[i]) * (iris.features[i] - means[i]);
        }
    }

    for (double& sigma : sigmas) {
        sigma = std::sqrt(sigma / dataset.size());
        if (sigma == 0) {
            std::cerr << "Standard deviation is zero for feature index " << (&sigma - &sigmas[0]) << std::endl;
            return;
        }
    }

    // Normalize the dataset
    for (auto& iris : dataset) {
        for (size_t i = 0; i < numFeatures; ++i) {
            iris.features[i] = (iris.features[i] - means[i]) / sigmas[i];
        }
    }
}

// Calculate Euclidean distance between two vectors
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// FCompute the nearest neighbor distance for each element within its class
std::vector<double> computeNearestNeighborDistances(const std::vector<Iris>& dataset) {
    std::unordered_map<std::string, std::vector<Iris> > classMap;
    std::vector<double> distances;
    
    // Group dataset by species
    for (const auto& iris : dataset) {
        classMap[iris.species].push_back(iris);
    }
    
    // Compute nearest neighbor distances for each class
    for (const auto& pair : classMap) {
        const auto& classData = pair.second;

        for (const auto& iris : classData) {
            double minDistance = std::numeric_limits<double>::max();
            for (const auto& neighbor : classData) {
                if (&iris != &neighbor) {
                    double distance = euclideanDistance(iris.features, neighbor.features);
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
            }
            // if the result of distance is bigger than 1, it will be dropped.
            if (minDistance <= 1) {
                distances.push_back(minDistance);
            }
        }
    }

    return distances;
}

int main() {
    std::string filename = "iris.data";
    std::vector<Iris> dataset = readDataset(filename);

    normalizeFeatures(dataset);

    std::vector<double> distances = computeNearestNeighborDistances(dataset);

    std::sort(distances.begin(), distances.end());
    distances.erase(unique(distances.begin(), distances.end()),distances.end());

    // Transform distances into the form [a]
    std::vector<std::vector<double> > formattedDistances;
    for (const double& distance : distances) {
        std::vector<double> distVec(1, distance); // Create a vector with one element
        formattedDistances.push_back(distVec);
    }

    size_t l = distances.size();
    std::vector<double> y(l);
    for (size_t i = 0; i < l; ++i) {
        y[i] = 1 - static_cast<double>(i + 1) / (l + 1);
    }

    // Print the distances for each element along with y
    /*std::cout << "Nearest neighbor distances (sorted) and corresponding y values:\n";
    for (size_t i = 0; i < l; ++i) {
        std::cout  << distances[i] << " " << y[i] << "\n";
    }*/

    // ALGLIB array initialization
    alglib::real_2d_array x;
    alglib::real_1d_array y_array;
    alglib::real_1d_array w;

    w.setlength(y.size());
    for(size_t i = 0; i < y.size(); i++) {
        w[i] = distances[i]*distances[i];  // Set each weight to 1
    }

    // Setting the length of the arrays
    x.setlength(formattedDistances.size(), 1);
    y_array.setlength(y.size());

    // Copying data from vector to ALGLIB array
    for(size_t i = 0; i < formattedDistances.size(); i++) {
        x[i][0] = formattedDistances[i][0];  // Assuming each subvector has exactly one element
    }

    for(size_t i = 0; i < y.size(); i++) {
        y_array[i] = y[i];
    }

    try
    {
        real_2d_array xv = x;
        real_1d_array yv = y_array;
        real_1d_array c = "[0.367, 0.45]";
        double epsx = 0;
        ae_int_t maxits = 0;
        lsfitstate state;
        lsfitreport rep;

        lsfitcreatewfg(xv, yv, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, sigf_a, sigf_ad);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - logistic(c[0], c[1], x[i][0]));
        }*/

        lsfitcreatewfg(xv, yv, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, sigf_b, sigf_bd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - Hyperbolic_tangent(c[0], c[1], x[i][0]));
        }*/

        lsfitcreatewfg(xv, yv, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, sigf_c, sigf_cd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - arctangent(c[0], c[1], x[i][0]));
        }*/

        lsfitcreatewfg(xv, yv, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, sigf_d, sigf_dd);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - gudermannian(c[0], c[1], x[i][0]));
        }*/
        
        lsfitcreatewfg(xv, yv, w, c, state);
        lsfitsetcond(state, epsx, maxits);
        alglib::lsfitfit(state, sigf_e, sigf_ed);
        lsfitresults(state, c, rep);
        //printf("%d\n", int(rep.terminationtype));
        printf("%s\n", c.tostring(1).c_str());
        printf("%f\n", rep.wrmserror);

        /*for (int i = 0; i < y.size(); i++){
            printf("xi: %g yi: %g f(%g,%g,xi): %g\n", x[i][0], y[i], c[0], c[1], 1 - algebraic(c[0], c[1], x[i][0]));
        }*/
    }
    catch(alglib::ap_error alglib_exception)
    {
        printf("ALGLIB exception with message '%s'\n", alglib_exception.msg.c_str());
        return 1;
    }

    return 0;
}