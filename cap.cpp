#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <unsupported/Eigen/NonLinearOptimization>

// Function to calculate mean and sigma of a feature
void calculateMeanAndSigma(const std::vector<double>& feature, double& mean, double& sigma) {
    double sum = 0.0;
    for (double value : feature) {
        sum += value;
    }
    mean = sum / feature.size();

    double sumSquaredDiff = 0.0;
    for (double value : feature) {
        double diff = value - mean;
        sumSquaredDiff += diff * diff;
    }
    sigma = std::sqrt(sumSquaredDiff / feature.size());
}

// Function to normalize a feature globally
void normalizeFeature(std::vector<double>& feature) {
    double mean, sigma;
    calculateMeanAndSigma(feature, mean, sigma);

    for (double& value : feature) {
        value = (value - mean) / sigma;
    }
}

// Function to compute distance between two elements
double computeDistance(const std::vector<double>& element1, const std::vector<double>& element2) {
    // Assuming Euclidean distance for simplicity
    double distance = 0.0;
    for (size_t i = 0; i < element1.size(); ++i) {
        distance += std::pow(element1[i] - element2[i], 2);
    }
    return std::sqrt(distance);
}

// Function to compute PDF and CDF
void computeDistanceDistribution(const std::vector<std::vector<double> >& classMembers) {
    std::vector<double> distances;

    for (size_t i = 0; i < classMembers.size(); ++i) {
        for (size_t j = 0; j < classMembers.size(); ++j) {
            if (i != j) {
                double distance = computeDistance(classMembers[i], classMembers[j]);
                distances.push_back(distance); // only one parameter to hold min distance
            }
        }
    }

    // Sort distances
    std::sort(distances.begin(), distances.end());

    // Compute PDF and CDF
    // need ecdf empirical cdf by sorting nearest distance from smallest to largest
    // i/n+1 
    // least square fit to find better interpolation
    size_t numDistances = distances.size();
    for (size_t i = 0; i < numDistances; ++i) {
        double pdf = 1.0 / numDistances;
        double cdf = static_cast<double>(i + 1) / numDistances;

        // Print or use PDF and CDF values
        std::cout << "Distance: " << distances[i] << ", PDF: " << pdf << ", CDF: " << cdf << std::endl;
    }
}

int main() {
    // Open the data file
    std::ifstream inputFile("glass+identification/glass.data");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    std::vector<std::vector<std::vector<double> > > datasets;
    
    // Read data from the file
    std::string line;
    int numClasses = -1;
    while (std::getline(inputFile, line)) {
        std::vector<double> dataPoint;
        std::istringstream iss(line);
        double value;
        char comma;
        int iclass;
        while (iss >> value) {
            iclass = value;
            dataPoint.push_back(value);
            iss >> comma;
        }
        if(iclass > numClasses) datasets[++numClasses] = {};
        datasets[iclass].push_back(dataPoint);
    }

    inputFile.close();

    // Normalization
    for (auto& dataset : datasets) {
        for (size_t i = 1; i < dataset[0].size() - 1; ++i) {
            std::vector<double> feature;
            for (const auto& dataPoint : dataset) {
                feature.push_back(dataPoint[i]);
            }
            normalizeFeature(feature);
            // Update the original dataset with normalized values
            for (size_t j = 0; j < dataset.size(); ++j) {
                dataset[j][i] = feature[j];
            }
        }
    }

    for (const auto& dataset : datasets) {
        // For each class, compute distance distribution and plot
        computeDistanceDistribution(dataset);
    }

    return 0;
}