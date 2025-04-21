#include <iostream>
#include <vector>
#include <unordered_map>
#include <array>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <iterator>

#include "fit.h"
#include "process.h"
#include "classMember.h"
#include "test.h"
#include "modelState.h"

using namespace std;

ModelState MODEL_STATE;
std::mutex m;
double NUM_THREADS;
int K_FOLDS = 10;

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

    size_t lineNumber = 2;
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
        obj.lineNumber = lineNumber++;
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
    size_t decimalPointCount = 0;
    for (size_t i = 0; i < std::strlen(value); ++i) {
        if (!isdigit(value[i])) {
            if (value[i] == '.' && decimalPointCount == 0) {
                ++decimalPointCount;
            }
            else {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Command line arguments can optionally include the p value threshold and write p values to a CSV
    /*
    double pvalueThreshold = -1;
    bool pValuesToCSV = false;
    std::string datasetFilename, pValuesCSVFilename;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pvalues_to_csv") == 0) {
            pValuesToCSV = true;
            ++i;
            if (i < argc - 1)
                pValuesCSVFilename = argv[i];
            else {
                std::cerr << "ERROR: Filename to write p values to was not provided.\n";
                return 0;
            }
        }
        else if (std::strcmp(argv[i], "-H") == 0) {
            std::cout << "Format: executable [--pvalues_to_csv filename] [pvalueThreshold] datasetFilepath\n";
            std::cout << "Example: ./cpv.exe --pvalues_to_csv result.csv 0.50 dataset.csv\n";
            return 0;
        }
        else if (i == argc - 2) {
            if (!isNonnegativeDouble(argv[i])) {
                std::cerr << "ERROR: pvalueThreshold must be nonnegative double.\n";
                return 0;
            }
            pvalueThreshold = std::atof(argv[i]);
        }
        else
            datasetFilename = argv[i];
    }

    if (datasetFilename == "") {
        std::cerr << "ERROR: No dataset provided.\n";
        return 0;
    }
    */

    bool bestFitFunctionsToCSV = std::atoi(argv[1]);
    std::string bestFitFunctionsCSVFilename = argv[2];
    bool pValuesToCSV = std::atoi(argv[3]);
    std::string pValuesCSVFilename = argv[4];
    bool summaryToCSV = std::atoi(argv[5]);
    std::string summaryCSVFilename = argv[6];
    bool applyPCA = std::atoi(argv[7]);
    double pvalueThreshold = (std::atof(argv[8]) > 0) ? std::atof(argv[8]) : -1;
    std::string datasetFilename = argv[9];

    std::vector<ClassMember> dataset = readFormattedDataset(datasetFilename);
    
    std::vector<ClassMember> kSets[K_FOLDS];
    kFoldSplit(dataset, kSets);
    
    setThreads();

    std::unordered_map<std::string, double[5]> predictionStatistics;
    std::unordered_map<std::string, double[3]> predictionStatisticsPerClass;
    std::unordered_map<std::string, double> numInstancesPerClass;

    for (int i = 0; i < K_FOLDS; ++i) {
        std::vector<ClassMember> trainDataset;
        
        for (int j = 0; j < K_FOLDS; ++j) {
            if (j != i) {
                trainDataset.reserve(trainDataset.size() + kSets[j].size());
                trainDataset.insert(trainDataset.end(), kSets[j].begin(), kSets[j].end());
            }
        }
        
        std::cout << "Iteration " << i << ":\n";

        std::unordered_map<std::string, std::vector<double> > sorted_distances = process(trainDataset, applyPCA);

        fitClasses(sorted_distances);

        test(kSets[i], predictionStatistics, predictionStatisticsPerClass, numInstancesPerClass, i, applyPCA,
            pvalueThreshold, bestFitFunctionsToCSV, bestFitFunctionsCSVFilename, pValuesToCSV, pValuesCSVFilename,
            summaryToCSV, summaryCSVFilename);
    }
}