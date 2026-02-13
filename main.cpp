#include <iostream>
#include <mutex>
#include <string>

#include "modelState.h"
#include "testResults.h"
#include "cachePaths.h"
#include "runFull.h"
#include "runFromCache.h"

ModelState MODEL_STATE;
TestResults TEST_RESULTS;
CachePaths CACHE_PATHS;
std::mutex m;
double NUM_THREADS;
int K_FOLDS = 10;
size_t MIN_CLASS_MEMBERS = 30;
size_t MAX_CLASS_MEMBERS = 1000;

void printBasicHelp() {
    std::string helpMsg = "Usage: ./cpv [-H] weightScheme pvalueThreshold featureWeighting neighborsChecked minSameClass "
        "datasetFilepath cacheDirectory\n"
        "\t -H: display more detailed help message\n"
        "\t weightScheme: {\"squared\", \"linear\", \"unweighted\", \"cube root\"}\n"
        "\t pvalueThreshold: {\"per class\"}\n"
        "\t featureWeighting: {0, 1}\n"
        "\t neighborsChecked: non negative integer\n"
        "\t minSameClass: non negative integer\n"
        "\t datasetFilepath: file containing dataset to run k fold cross validation on\n"
        "\t cacheDirectory: folder where output logs from training and testing will be stored\n";

    std::cout << helpMsg;
}

void printDetailedHelp() {
    std::string helpMsg = "Usage: ./cpv [-H] weightScheme pvalueThreshold featureWeighting neighborsChecked minSameClass "
        "datasetFilepath cacheDirectory\n"
        "\t -H: Display more detailed help message.\n"
        "\t weightScheme {\"squared\", \"linear\", \"unweighted\", \"cube root\"}: Weight assigned to points in the "
        "empirical distribution curve (ECDF) during curve fitting, as a function of nearest neighbor distance. "
        "Recommended to choose \"squared.\"\n"
        "\t pvalueThreshold {\"per class\"}\n"
        "\t featureWeighting {0, 1}: Boolean value for whether to apply feature weighting during nearest neighbor "
        "distance calculations.\n"
        "\t neighborsChecked {non negative integer}: Number of global nearest neighbors to find for \"voting off the "
        "island.\" 5 is generally a good choice.\n"
        "\t minSameClass {non negative integer}: Minimum number of global nearest neighbors required to keep a datapoint "
        "during \"voting off the island.\" 3 is generally a good choice. 0 is the equivalent of disabling \"voting off the "
        "island,\" and it is recommended to also pass 0 for neighborsChecked in this case.\n"
        "\t datasetFilepath: File containing dataset to run k fold cross validation on. The first row must be the "
        "header; the first column must be the class column; and subsequent columns must be the features. Prepending "
        "\"nonNum\" to a column name will cause that column and all subsequent columns to be disregarded.\n"
        "\t cacheDirectory: Folder where output logs from training and testing will be stored.\n\n"
        "Recommendation: Redirect the standard output to a file to save additional helpful information, including "
        "confusion matrices for the classes and \"none of the above\" (NOTA) points.\n";

    std::cout << helpMsg;
}

int main(int argc, char* argv[]) {
    switch (argc) {
        case 4:
            runFromPValues(argv);
            break;
        case 5:
            runFromNNDistances(argv);
            break;
        case 8:
            runFull(argc, argv);
            break;
        case 10:
            runFull(argc, argv);
            break;
        default:
            if (argc > 1 && std::strcmp(argv[1], "-H") == 0) {
                printDetailedHelp();
            }
            else {
                printBasicHelp();
            }
    } 
}
