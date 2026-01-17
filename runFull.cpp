#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>

#include "runFull.h"
#include "classMember.h"
#include "modelState.h"
#include "testResults.h"
#include "cachePaths.h"
#include "process.h"
#include "fit.h"
#include "test.h"
#include "saveResults.h"
#include "NOTAPoints.h"

std::unordered_map<std::string, std::vector<ClassMember> > createTrainDataset(
    std::unordered_map<std::string, std::vector<ClassMember> > kSets[], size_t fold) {
    std::unordered_map<std::string, std::vector<ClassMember> > trainDataset;
    size_t pointsPerFold = MAX_CLASS_MEMBERS / (K_FOLDS - 1);
    size_t remainder = MAX_CLASS_MEMBERS - (pointsPerFold * (K_FOLDS - 1));
    std::random_device rand_d;
    std::mt19937 gen(rand_d());

    for (const std::string& className : MODEL_STATE.classNames) {
        std::vector<ClassMember>& classTrainData = trainDataset[className];
        size_t availablePoints = MODEL_STATE.numInstancesPerClass.at(className) - kSets[fold].at(className).size();
        if (availablePoints > MAX_CLASS_MEMBERS) {
            classTrainData.reserve(MAX_CLASS_MEMBERS);
            size_t extraDatapoints = remainder;
            for (size_t i = 0; i < K_FOLDS; ++i) {
                if (i != fold) {
                    std::vector<ClassMember>& classData = kSets[i].at(className);
                    size_t numInsertPoints = ((classData.size() == pointsPerFold) || (extraDatapoints == 0)) ? pointsPerFold : pointsPerFold + 1;
                    if (classData.size() > numInsertPoints) {
                        std::shuffle(classData.begin(), classData.end(), gen);
                    }
                    if (extraDatapoints > 0) {
                        --extraDatapoints;
                    }
                    classTrainData.insert(classTrainData.end(), classData.begin(), classData.begin() + numInsertPoints);
                }
            }
        }
        else {
            classTrainData.reserve(availablePoints);
            for (size_t i = 0; i < K_FOLDS; ++i) {
                if (i != fold) {
                    std::vector<ClassMember>& classData = kSets[i].at(className);
                    classTrainData.insert(classTrainData.end(), classData.begin(), classData.end());
                }
            }
        }
    }

    return trainDataset;
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

std::vector<ClassMember> createTestSet(std::unordered_map<std::string, std::vector<ClassMember> > kSets[],
    const std::vector<ClassMember>& injectedPoints, size_t fold) {
    std::vector<ClassMember> testDataset;

    // Add datapoints from the current fold to the test set
    for (const auto& pair : kSets[fold]) {
        testDataset.reserve(testDataset.size() + pair.second.size());
        testDataset.insert(testDataset.end(), pair.second.begin(), pair.second.end());
    }

    // Add "randomly placed" points to the test set
    testDataset.reserve(testDataset.size() + injectedPoints.size());
    testDataset.insert(testDataset.end(), injectedPoints.begin(), injectedPoints.end());

    return testDataset;
}

void addSigmoidFit(double* bestFitData, const std::string& className, const std::string& functionName) {
    FitResult classSigmoid;
    classSigmoid.c.setlength(2);
    classSigmoid.c[0] = bestFitData[0];
    classSigmoid.c[1] = bestFitData[1];
    classSigmoid.functionName = functionName;
    classSigmoid.wrmsError = bestFitData[2];
    MODEL_STATE.bestFit[className] = std::move(classSigmoid);
}

void setSigmoidFitsFromFile(const std::string& sigmoidFitsFilename) {
    std::ifstream sigmoidFitsFile(sigmoidFitsFilename);
    std::stringstream ss;
    std::string line, field, className, functionName;
    std::unordered_map<std::string, std::unordered_map<std::string, double[3]> > bestFits;
    std::unordered_map<std::string, std::unordered_map<std::string, size_t> > bestFitCount;

    // Check whether the sigmoid fit data is per fold or for the overall class
    std::getline(sigmoidFitsFile, line);
    bool includesFold = line.rfind("fold", 0) != std::string::npos;

    // Read all sigmoid fit data
    while (std::getline(sigmoidFitsFile, line)) {
        ss.str("");
        ss.clear();
        ss << line;
        if (includesFold) {
            std::getline(ss, field, ',');
        }
        std::getline(ss, className, ',');
        std::getline(ss, functionName, ',');

        for (size_t i = 0; i < 3; ++i) {
            std::getline(ss, field, ',');
            bestFits[className][functionName][i] += std::stod(field);
        }

        ++bestFitCount[className][functionName];
    }

    sigmoidFitsFile.close();

    for (const auto& pair : bestFitCount) {
        if (includesFold) {
            // If the sigmoid fit data is per fold, calculate the overall parameter values for the class
            auto mostFrequentFit = std::max_element(pair.second.begin(), pair.second.end(),
                [](const std::pair<std::string, size_t>& fit1, const std::pair<std::string, size_t>& fit2) {
                    return fit1.second < fit2.second;
                });
            double* bestFitData = bestFits[pair.first][mostFrequentFit->first];

            for (size_t i = 0; i < 3; ++i) {
                bestFitData[i] /= mostFrequentFit->second;
            }

            addSigmoidFit(bestFitData, pair.first, mostFrequentFit->first);
        }
        else {
            auto bestFit = bestFits[pair.first].begin();
            addSigmoidFit(bestFit->second, pair.first, bestFit->first);
        }        
    }

    if (includesFold) {
        writeBestFitFunctionsToCSV(CACHE_PATHS.bestFitFunctionsFilepath);
    }
}

void runFull(int argc, char* argv[]) {
    std::string weightScheme = argv[1];
    std::string pValueThreshold = argv[2];
    bool applyFeatureWeighting = std::atoi(argv[3]);
    int numNeighborsChecked = std::atoi(argv[4]);
    int minSameClassCount = std::atoi(argv[5]);

    std::string sigmoidFitsFilename, NOTAPointsFilename, datasetFilename, cacheDirectory;
    if (argc == 10) {
        sigmoidFitsFilename = argv[6];
        NOTAPointsFilename = argv[7];
        datasetFilename = argv[8];
        cacheDirectory = argv[9];
    }
    else {
        datasetFilename = argv[6];
        cacheDirectory = argv[7];
    }

    std::unordered_map<std::string, std::vector<ClassMember> > dataset = readFormattedDataset(datasetFilename);
    std::unordered_map<std::string, std::vector<ClassMember> > kSets[K_FOLDS];
    kFoldSplit(dataset, kSets, 1000);

    CACHE_PATHS.initPaths(cacheDirectory);
    setThreads();

    // Set what type of processing will occur
    if (applyFeatureWeighting) {
        MODEL_STATE.processType = "featureWeighting";
    }
    else {
        MODEL_STATE.processType = "default";
    }

    MODEL_STATE.setDatasetSize();
    MODEL_STATE.setWeightExp(weightScheme);
    setPValueThreshold(pValueThreshold);
    MODEL_STATE.preexistingBestfit = fileExists(sigmoidFitsFilename);

    std::vector<ClassMember> NOTAPoints = (fileExists(NOTAPointsFilename)) ? readNOTAPointsFromFile(NOTAPointsFilename)
        : createNOTAPoints(dataset);

    if (MODEL_STATE.preexistingBestfit) {
        setSigmoidFitsFromFile(sigmoidFitsFilename);
    }

    for (int fold = 0; fold < K_FOLDS; ++fold) {
        std::unordered_map<std::string, std::vector<ClassMember> > trainDataset;

        for (int j = 0; j < K_FOLDS; ++j) {
            if (j != fold) {
                for (const auto& pair : kSets[j]) {
                    std::vector<ClassMember>& classData = trainDataset[pair.first];
                    classData.reserve(classData.size() + pair.second.size());
                    classData.insert(classData.end(), pair.second.begin(), pair.second.end());
                }
            }
        }
        //std::unordered_map<std::string, std::vector<ClassMember> > trainDataset = createTrainDataset(kSets, fold);

        std::cout << "Iteration " << fold << ":\n";

        std::cout << "Original training dataset sizes:\n";
        int totalDatasetPoints = 0;
        for (const auto& pair : trainDataset) {
            std::cout << "Class " << pair.first << ": " << pair.second.size() << " instances\n";
            totalDatasetPoints += pair.second.size();
        }
        std::cout << "Total in dataset: " << totalDatasetPoints << "\n";

        std::unordered_map<std::string, std::vector<double> > sorted_distances = process(trainDataset,
            numNeighborsChecked, minSameClassCount, fold);

        // Print out difference in dataset sizes between the original and filtered training dataset
        std::cout << "\nFiltered training dataset sizes:\n";
        int totalFilteredPoints = 0;
        for (const auto& pair : MODEL_STATE.classMap) {
            std::cout << "Class " << pair.first << ": " << pair.second.size() << " instances\n";
            totalFilteredPoints += pair.second.size();
        }
        std::cout << "Total in filteredDataset: " << totalFilteredPoints << "\n\n";

        if (!MODEL_STATE.preexistingBestfit) {
            fitClasses(sorted_distances, fold);
        }

        std::vector<ClassMember> testDataset = createTestSet(kSets, NOTAPoints, fold);

        test(testDataset, fold);

        MODEL_STATE.clearTemporaries();
    }
}
