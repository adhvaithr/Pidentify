#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <thread>
#include <mutex>
#include <algorithm>

#include "fit.h"
#include "process.h"
#include "classMember.h"
#include "test.h"
#include "modelState.h"

using namespace std;

ModelState MODEL_STATE;
std::mutex m;
double NUM_THREADS;

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

// Initialize number of threads to use concurrently
void setThreads() {
    NUM_THREADS = std::max(static_cast<int>(std::thread::hardware_concurrency()), 4);
}

bool isNonnegativeDouble(char* value) {
    for (size_t i = 0; i < std::strlen(value); ++i) {
        if (!isdigit(value[i]) && value[i] != '.') {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Error checking on command line arguments
    if ((argc < 4) || (argc > 5)) {
        std::cout << "Incorrect number of command line arguments passed.\n";
        std::cout << "Format: executable pvalueThreshold testSize datasetFilepath [testDatasetFilepath]\n";
        std::cout << "Example: ./cpv.exe 0.50 0.30 filename.csv\n";
        return 0;
    }
    
    if (!isNonnegativeDouble(argv[1]) || !isNonnegativeDouble(argv[2])) {
        std::cout << "pvalueThreshold and testSize must be nonnegative doubles.\n";
        return 0;
    }

    double pvalueThreshold = std::atof(argv[1]), testSize = std::atof(argv[2]);
    std::string datasetFilename = argv[3], testDatasetFilename;
    if (argc == 5) { testDatasetFilename = argv[4]; }

    setThreads();

    std::vector<ClassMember> dataset = readFormattedDataset(datasetFilename);

    std::vector<ClassMember> testDataset;
    if (testDatasetFilename != "") {
        testDataset = readFormattedDataset(testDatasetFilename);
    }
    else {
        trainTestSplit(dataset, testDataset, testSize);
    }
    
    std::unordered_map<std::string, std::vector<double> > sorted_distances = process(dataset);

    fitClasses(sorted_distances);

    test(testDataset, pvalueThreshold);
}