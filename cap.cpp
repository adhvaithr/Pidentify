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
#include "curve_fitting.h"
#include "data_processing.h"
#include "iris.h"

using namespace std;

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

int main() {
    std::string filename = "iris.data";
    std::vector<Iris> dataset = readDataset(filename);

    std::vector<double> sorted_distances = process(dataset);

    size_t l = sorted_distances.size();
    std::vector<double> y(l);
    for (size_t i = 0; i < l; ++i) {
        y[i] = 1 - static_cast<double>(i + 1) / (l + 1);
    }

    curve_fitting(sorted_distances, y);
}