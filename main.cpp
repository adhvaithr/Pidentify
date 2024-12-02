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

/*Read dataset from custom formatted file where columns are in the following order:
class name, numerical features, nonnumerical features (if any).*/
std::vector<ClassMember> readFormattedDataset(const std::string& filename) {
    std::vector<ClassMember> dataset;
    std::ifstream file(filename);
    std::string line;

    // Read the header
    std::getline(file, line);
    std::stringstream header(line);
    std::string colName;
    int numFeatures = 0;
    std::getline(header, colName, ',');
    // Count the number of numerical features
    while (std::getline(header, colName, ',')) {
        if (colName.compare(0, 6, "nonNum") == 0) {
            break;
        }
        numFeatures += 1;
    }

    // Only add the numerical features to the ClassMember vector
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ClassMember obj;
        std::string feature;
        std::getline(ss, obj.name, ',');
        for (int i = 0; i < numFeatures; ++i) {
            std::getline(ss, feature, ',');
            obj.features.push_back(std::stod(feature));
        }

        dataset.push_back(obj);
    }
    file.close();

    return dataset;
}

int main(int argc, char* argv[]) {
    std::string filename = argv[1];
    std::vector<ClassMember> dataset = readFormattedDataset(filename);

    std::vector<double> sorted_distances = process(dataset);

    size_t l = sorted_distances.size();

    // consturct corresponding y values in terms of distances for ECDF points
    std::vector<double> y(l);
    for (size_t i = 0; i < l; ++i) {
        y[i] = 1 - static_cast<double>(i + 1) / (l + 1);
    }

    curveFitting(sorted_distances, y);
}