#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>

#include "fit.h"
#include "process.h"
#include "classMember.h"

using namespace std;

// Read the dataset from a file
std::vector<ClassMember> readDataset(const std::string& filename) {
    std::vector<ClassMember> dataset;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;  // Skip the empty lines
        }

        std::stringstream ss(line);
        ClassMember obj;
        std::string feature;

        while (std::getline(ss, feature, ',')) {
            if (isdigit(feature[0]) || feature[0] == '-') {
                obj.features.push_back(std::stod(feature));
            } else {
                assert(obj.name == "");
                assert(feature != "");
                obj.name = feature;
            }
        }
        dataset.push_back(obj);
    }

    //std::cout << "feature size:" << dataset[0].features.size() << std::endl;
    return dataset;
}

int main() {
    std::string filename = "iris.data";
    std::vector<ClassMember> dataset = readDataset(filename);

    std::vector<double> sorted_distances = process(dataset);

    size_t l = sorted_distances.size();

    // consturct corresponding y values in terms of distances for ECDF points
    std::vector<double> y(l);
    for (size_t i = 0; i < l; ++i) {
        y[i] = 1 - static_cast<double>(i + 1) / (l + 1);
    }

    curveFitting(sorted_distances, y);
}