#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include "runFromCache.h"
#include "modelState.h"
#include "cachePaths.h"
#include "classMember.h"
#include "testResults.h"
#include "test.h"
#include "fit.h"

void reinitSS(std::stringstream& ss, const std::string& newContents) {
    ss.str("");
    ss.clear();
    ss << newContents;
}

// Populate ClassMember objects with the class name; fill in nearest neighbor distances for each fold; create
// tallies for the number of instances in each class, stored in MODEL_STATE, and total number of randomly placed
// points across all folds, stored in TEST_RESULTS; and populate all class names, stored in MODEL_STATE
void readClassificationsDirectory(const std::string& classificationsDirectory,
    std::unordered_map<size_t, std::vector<ClassMember> >& dataset,
    std::unordered_map<size_t, std::vector<std::unordered_map<std::string, double> > >& nnDistances) {
    std::string pathSep = getPathSep();
    std::stringstream ss;
    std::string line, value;

    for (int fold = 0; fold < K_FOLDS; ++fold) {
        std::vector<std::string> foldClasses;
        std::ifstream classificationsFile(classificationsDirectory + pathSep + "classifications-iter"
            + std::to_string(fold) + ".csv");

        // Get all the classes that were fit for that fold
        std::getline(classificationsFile, line);
        reinitSS(ss, line);
        // Ignore the first two column names in the header
        for (size_t i = 0; i < 2; ++i) {
            std::getline(ss, value, ',');
        }
        // Record all the classes fit for that fold
        while (std::getline(ss, value, ',')) {
            foldClasses.push_back(value);
        }
        size_t numFoldClasses = foldClasses.size();

        // Get the nearest neighbor distances for all the datapoints in the fold
        while (std::getline(classificationsFile, line)) {
            reinitSS(ss, line);
            ClassMember obj;
            std::getline(ss, obj.name, ',');
            std::getline(ss, value, ','); // Ignore the class the datapoint was in from when the file was created
            std::getline(ss, value, ',');
            // Get all the nearest neighbor distances for the current datapoint
            nnDistances[fold].emplace_back(std::unordered_map<std::string, double>{ { foldClasses[0], std::stod(value) } });
            size_t idx = nnDistances[fold].size() - 1;
            for (size_t i = 1; i < numFoldClasses; ++i) {
                std::getline(ss, value, ',');
                nnDistances[fold][idx][foldClasses[i]] = std::stod(value);
            }
            if (obj.name == "NOTA") {
                //obj.NOTA = true;
                //++TEST_RESULTS.randomPoints[0];
            }
            else {
                ++MODEL_STATE.numInstancesPerClass[obj.name];
            }
            dataset[fold].push_back(obj);
        }

        classificationsFile.close();
    }

    // Get all class names in alphabetical order
    for (const auto& pair : MODEL_STATE.numInstancesPerClass) {
        MODEL_STATE.classNames.push_back(pair.first);
    }
    std::sort(MODEL_STATE.classNames.begin(), MODEL_STATE.classNames.end());
}

// Get the p values for each datapoint in the dataset
std::unordered_map<size_t, std::vector<std::unordered_map<std::string, double> > > readPValuesFile(
    const std::string& pValuesFilepath) {
    std::unordered_map<size_t, std::vector<std::unordered_map<std::string, double> > > pvalues;
    std::ifstream pValuesFile(pValuesFilepath);
    std::stringstream ss;
    std::string line, value;
    size_t fold;

    // Ignore the header
    std::getline(pValuesFile, line);

    while (std::getline(pValuesFile, line)) {
        reinitSS(ss, line);
        std::getline(ss, value, ','); // Ignore the line number of the datapoint
        std::getline(ss, value, ',');
        fold = std::stoul(value);
        // Get all the p values for the current datapoint
        std::getline(ss, value, ',');
        pvalues[fold].emplace_back(std::unordered_map<std::string, double>{ { MODEL_STATE.classNames[0], std::stod(value) } });
        size_t idx = pvalues[fold].size() - 1;
        for (auto iter = MODEL_STATE.classNames.begin() + 1; iter != MODEL_STATE.classNames.end(); ++iter) {
            std::getline(ss, value, ',');
            pvalues[fold][idx][*iter] = std::stod(value);
        }
    }

    pValuesFile.close();

    return pvalues;
}

// Get the x- and y- coordinates in the ECDF
void readECDFFile(const std::string& ecdfFilename,
    std::unordered_map<std::string, std::vector<double> >& sorted_distances,
    std::unordered_map<std::string, std::vector<double> >& y_values) {
    std::ifstream ecdfFile(ecdfFilename);
    std::stringstream ss;
    std::string line, value, className;

    // Ignore the header
    std::getline(ecdfFile, line);

    while (std::getline(ecdfFile, line)) {
        reinitSS(ss, line);
        std::getline(ss, className, ',');
        std::getline(ss, value, ',');
        sorted_distances[className].push_back(std::stod(value));
        std::getline(ss, value);
        y_values[className].push_back(std::stod(value));
    }

    ecdfFile.close();
}

// Run starting with the nearest neighbor distances, useful for testing different weight schemes
void runFromNNDistances(char* argv[]) {
    std::string weightScheme = argv[1];
    std::string pValueThreshold = argv[2];
    std::string prevResultsDirectory = argv[3];
    std::string cacheDirectory = argv[4];

    CACHE_PATHS.initPaths(cacheDirectory);
    CachePaths PrevResultsPaths;
    PrevResultsPaths.initPaths(prevResultsDirectory);
    setThreads();

    std::string pathSep = getPathSep();
    std::unordered_map<size_t, std::vector<ClassMember> > dataset;
    std::unordered_map<size_t, std::vector<std::unordered_map<std::string, double> > > nnDistances;

    readClassificationsDirectory(PrevResultsPaths.classificationsDirectory, dataset, nnDistances);

    MODEL_STATE.setDatasetSize();
    MODEL_STATE.setWeightExp(weightScheme);
    setPValueThreshold(pValueThreshold);

    for (int fold = 0; fold < K_FOLDS; ++fold) {
        std::cout << "Iteration " << fold << ":\n";

        std::string ecdfFilepath = PrevResultsPaths.ecdfDirectory + pathSep + "iter" + std::to_string(fold) + ".csv";
        std::unordered_map<std::string, std::vector<double> > sorted_distances;
        std::unordered_map<std::string, std::vector<double> > y_values;

        readECDFFile(ecdfFilepath, sorted_distances, y_values);
        
        fitClasses(sorted_distances, y_values);
        test(dataset[fold], nnDistances[fold], fold);
        MODEL_STATE.clearTemporaries();
    }
}

// Run starting with the p values, useful for testing different p value thresholds
void runFromPValues(char* argv[]) {
    std::string pValueThreshold = argv[1];
    std::string prevResultsDirectory = argv[2];
    std::string cacheDirectory = argv[3];

    CACHE_PATHS.initPaths(cacheDirectory);
    CachePaths PrevResultsPaths;
    PrevResultsPaths.initPaths(prevResultsDirectory);

    std::unordered_map<size_t, std::vector<ClassMember> > dataset;
    std::unordered_map<size_t, std::vector<std::unordered_map<std::string, double> > > nnDistances;

    readClassificationsDirectory(PrevResultsPaths.classificationsDirectory, dataset, nnDistances);
    std::unordered_map<size_t, std::vector<std::unordered_map<std::string, double> > > pvalues = readPValuesFile(
        PrevResultsPaths.pvaluesFilepath);

    MODEL_STATE.setDatasetSize();
    setPValueThreshold(pValueThreshold);

    for (int fold = 0; fold < K_FOLDS; ++fold) {
        test(dataset[fold], pvalues[fold], nnDistances[fold], fold);
    }
}
