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
#include <random>
#include <chrono>
#include <cmath>
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

/*Limit number of instances in each class */
std::vector<ClassMember> trimmedDataset(const std::vector<ClassMember>& dataset, size_t maxPerClass = 1000) {
    std::unordered_map<std::string, std::vector<ClassMember>> classCountMap;
    for (const auto& member : dataset) {
        classCountMap[member.name].push_back(member);
    }

    std::vector<ClassMember> trimmed;
    std::random_device rand_d;

    std::mt19937 gen(rand_d());

    for (auto& currClass : classCountMap) {
        auto& members = currClass.second;
        if (members.size() > maxPerClass) {
            std::shuffle(members.begin(), members.end(), gen);
            trimmed.insert(trimmed.end(), members.begin(), members.begin() + maxPerClass);
        } else {
            trimmed.insert(trimmed.end(), members.begin(), members.end());
        }
    }
    return trimmed;
}

void distributeAcrossFolds(const std::unordered_map<std::string, std::vector<ClassMember> >::const_iterator& datasetIter, std::unordered_map<std::string, std::vector<ClassMember> > kSets[]) {
    auto& datapoints = datasetIter->second;
    size_t minDatapointsPerFold = datapoints.size() / K_FOLDS;

    size_t extraDatapoints = datapoints.size() - minDatapointsPerFold * K_FOLDS;
    std::vector<ClassMember>::const_iterator datapointsIter = datapoints.begin();

    for (int i = 0; i < K_FOLDS; ++i) {
        size_t datapointsToInsert = (i < extraDatapoints) ? minDatapointsPerFold + 1 : minDatapointsPerFold;
        std::vector<ClassMember>& foldClassData = kSets[i][datasetIter->first];
        foldClassData.reserve(datapointsToInsert);
        foldClassData.insert(foldClassData.end(), datapointsIter, datapointsIter + datapointsToInsert);
        datapointsIter += datapointsToInsert;
    }
}

void kFoldSplit(std::unordered_map<std::string, std::vector<ClassMember> >& dataset,
    std::unordered_map<std::string, std::vector<ClassMember> > kSets[], size_t maxPerClass = 1000) {
    std::random_device rand_d;

    std::mt19937 gen(rand_d());

    for (auto iter = dataset.begin(); iter != dataset.end();) {
        auto& members = iter->second;
        if (members.size() < K_FOLDS) {
            iter = dataset.erase(iter);
            continue;
        }
        std::shuffle(members.begin(), members.end(), gen);
        if (members.size() > maxPerClass) {
            members.resize(maxPerClass);
        }
        distributeAcrossFolds(iter, kSets);
        MODEL_STATE.classNames.push_back(iter->first);
        ++iter;
    }

    std::sort(MODEL_STATE.classNames.begin(), MODEL_STATE.classNames.end());
}

std::unordered_map<std::string, std::vector<ClassMember> > readFormattedDataset(const std::string& filename) {
    std::unordered_map<std::string, std::vector<ClassMember> > dataset;
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
        dataset[obj.name].push_back(obj);
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
   bool applyFeatureWeighting = std::atoi(argv[8]);
   double pvalueThreshold = (std::atof(argv[9]) > 0) ? std::atof(argv[9]) : -1;
   int numNeighborsChecked = std::atoi(argv[10]);
   int minSameClassCount = std::atoi(argv[11]);
   std::string datasetFilename = argv[12];
  

    std::unordered_map<std::string, std::vector<ClassMember> > dataset = readFormattedDataset(datasetFilename);
    std::unordered_map<std::string, std::vector<ClassMember> > kSets[K_FOLDS];
    kFoldSplit(dataset, kSets, 1000);

    setThreads();


    std::unordered_map<std::string, double[5]> predictionStatistics;
    std::unordered_map<std::string, double[3]> predictionStatisticsPerClass;
    std::unordered_map<std::string, double> numInstancesPerClass;


    // Set what type of processing will occur
    if (applyPCA && dataset.begin()->second[0].features.size() >= 3) {
        MODEL_STATE.processType = "PCA";
    }
    else if (applyFeatureWeighting) {
        MODEL_STATE.processType = "featureWeighting";
    }
    else {
        MODEL_STATE.processType = "default";
    }

    for (int i = 0; i < K_FOLDS; ++i) {
        std::unordered_map<std::string, std::vector<ClassMember> > trainDataset;
        
        for (int j = 0; j < K_FOLDS; ++j) {
            if (j != i) {
                for (const auto& pair : kSets[j]) {
                    std::vector<ClassMember>& classData = trainDataset[pair.first];
                    classData.reserve(classData.size() + pair.second.size());
                    classData.insert(classData.end(), pair.second.begin(), pair.second.end());
                }
            }
        }
        
        std::cout << "Iteration " << i << ":\n";
        std::unordered_map<std::string, std::vector<ClassMember> > filteredDataset;
        std::unordered_map<std::string, std::vector<double> > sorted_distances = process(trainDataset, filteredDataset,numNeighborsChecked, minSameClassCount);


        fitClasses(sorted_distances);


        std::vector<ClassMember> testDataset;
        for (const auto& pair : kSets[i]) {
            for (const auto& obj : pair.second) {
                testDataset.push_back(obj);
            }
        }


        test(testDataset, predictionStatistics, predictionStatisticsPerClass, numInstancesPerClass, i,
            pvalueThreshold, bestFitFunctionsToCSV, bestFitFunctionsCSVFilename, pValuesToCSV, pValuesCSVFilename,
            summaryToCSV, summaryCSVFilename);
        
        std::cout << "Original dataset sizes:\n";
        int totalDatasetPoints = 0;
        for (const auto& pair : dataset) {
            std::cout << "Class " << pair.first << ": " << pair.second.size() << " instances\n";
            totalDatasetPoints += pair.second.size();
        }
        std::cout << "Total in dataset: " << totalDatasetPoints << "\n";


        std::cout << "\nFiltered dataset sizes:\n";
        int totalFilteredPoints = 0;
        for (const auto& pair : filteredDataset) {
            std::cout << "Class " << pair.first << ": " << pair.second.size() << " instances\n";
            totalFilteredPoints += pair.second.size();
        }
        std::cout << "Total in filteredDataset: " << totalFilteredPoints << "\n";


    }
}





