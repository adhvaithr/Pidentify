#include <iostream>
#include <mutex>

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

int main(int argc, char* argv[]) {
    switch (argc) {
        case 4:
            runFromPValues(argv);
            break;
        case 5:
            runFromNNDistances(argv);
            break;
        case 8:
            runFull(argv);
            break;
    } 
}
