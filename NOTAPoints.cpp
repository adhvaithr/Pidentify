#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "nanoflann/KDTreeVectorOfVectorsAdaptor.h"

#include "NOTAPoints.h"
#include "modelState.h"
#include "testResults.h"
#include "process.h"
#include "cachePaths.h"

void updateFeatureBB(const std::vector<double>& features) {
	size_t dim = features.size();

	for (size_t i = 0; i < dim; ++i) {
		if (features[i] < MODEL_STATE.featureMins[i]) {
			MODEL_STATE.featureMins[i] = features[i];
		}
		if (features[i] > MODEL_STATE.featureMaxs[i]) {
			MODEL_STATE.featureMaxs[i] = features[i];
		}
	}
}

void findFeatureBB(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset, double extension) {
	MODEL_STATE.featureMins = MODEL_STATE.featureMaxs = dataset.begin()->second[0].features;
	size_t dim = dataset.begin()->second[0].features.size();

	for (const auto& pair : dataset) {
		for (const auto& obj : pair.second) {
			updateFeatureBB(obj.features);
		}
	}

	for (size_t i = 0; i < dim; ++i) {
		double extensionVal = (MODEL_STATE.featureMaxs[i] - MODEL_STATE.featureMins[i]) * extension;
		MODEL_STATE.featureMins[i] -= extensionVal;
		MODEL_STATE.featureMaxs[i] += extensionVal;
	}
}

// Convert integer index into the multiplier for void NOTA points
double NNStepMultiplier(size_t idx) {
	return NN_STEPS_START + (NN_STEP_SIZE * idx);
}

// Convert multiplier into integer index for void NOTA points or -1 for hyperspace NOTA points
int NNUnitsFromClass(double multiplier) {
	if (multiplier < NN_STEPS_START || multiplier > NN_STEPS_START + NN_STEP_SIZE * (NUM_NN_STEPS - 1)) {
		return -1;
	}
	else {
		return std::round((multiplier - NN_STEPS_START) / NN_STEP_SIZE);
	}
}

/*
void insertVoidNOTAPoints(const std::vector<double>& featureMins, const std::vector<double>& featureMaxs, size_t idx,
	std::mt19937& gen,
	const std::unordered_map<std::string, KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double>* >& kdTrees,
	const std::vector<std::unordered_map<std::string, double> >& minNNDistances, std::vector<ClassMember>& NOTAPoints) {
	size_t total = std::max(MODEL_STATE.datasetSize, MIN_NOTA_POINTS);
	//size_t total = 10;
	size_t dim = featureMins.size();
	size_t neighborIndices[1];
	double squaredDistances[1];

	// Create random number generators
	std::vector<std::uniform_real_distribution<double> > featureGenerators(dim);
	for (size_t i = 0; i < dim; ++i) {
		featureGenerators[i] = std::move(std::uniform_real_distribution<double>(featureMins[i], featureMaxs[i]));
	}

	size_t created = 0;
	bool validNOTAPoint;

	for (size_t attempts = 0, maxAttempts = total * 10; created < total && attempts < maxAttempts; ++attempts) {
		// Create a new datapoint
		ClassMember NOTAPoint(std::vector<double>(dim), "NOTA", 0, NOTACategory::VOID, idx);
		for (size_t i = 0; i < dim; ++i) {
			NOTAPoint.features[i] = featureGenerators[i](gen);
		}

		// Check if datapoint satisfies the minimum nearest neighbor distance from classes requirement
		validNOTAPoint = true;
		for (const auto& kdTree : kdTrees) {
			kdTree.second->query(&NOTAPoint.features[0], 1, &neighborIndices[0], &squaredDistances[0]);
			if (std::sqrt(squaredDistances[0]) < minNNDistances[idx].at(kdTree.first)) {
				validNOTAPoint = false;
				break;
			}
		}

		// Add datapoint to NOTA points if it satisfies minimum nearest neighbor distance requirement
		if (validNOTAPoint) {
			NOTAPoints.push_back(std::move(NOTAPoint));
			++created;
		}
	}

	// Update the state to reflect the total void NOTA points that will be tested on across all folds
	TEST_RESULTS.voidRandomPoints[idx][0] = created * K_FOLDS;
}
*/

void insertVoidNOTAPoints(const std::vector<double>& featureMins, const std::vector<double>& featureMaxs, size_t idx,
	std::mt19937& gen,
	const std::unordered_map<std::string, KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double>* >& kdTrees,
	const std::vector<std::unordered_map<std::string, double> >& minNNDistances, std::vector<ClassMember>& NOTAPoints) {
	size_t total = std::max(MODEL_STATE.datasetSize, MIN_NOTA_POINTS);
	//size_t total = 10;
	size_t dim = featureMins.size();
	size_t neighborIndices[1];
	double squaredDistances[1];

	// Create random number generators
	std::vector<std::uniform_real_distribution<double> > featureGenerators(dim);
	for (size_t i = 0; i < dim; ++i) {
		featureGenerators[i] = std::move(std::uniform_real_distribution<double>(featureMins[i], featureMaxs[i]));
	}

	size_t created = 0;
	bool validNOTAPoint;

	for (size_t attempts = 0, maxAttempts = total * 10; created < total && attempts < maxAttempts; ++attempts) {
		// Create a new datapoint
		ClassMember NOTAPoint(std::vector<double>(dim), "NOTA", 0, NOTACategory::VOID, idx);
		for (size_t i = 0; i < dim; ++i) {
			NOTAPoint.features[i] = featureGenerators[i](gen);
		}

		// Check if datapoint satisfies the minimum nearest neighbor distance from classes requirement
		validNOTAPoint = true;
		for (const auto& kdTree : kdTrees) {
			kdTree.second->query(&NOTAPoint.features[0], 1, &neighborIndices[0], &squaredDistances[0]);
			if (std::sqrt(squaredDistances[0]) < minNNDistances[idx].at(kdTree.first)) {
				validNOTAPoint = false;
				break;
			}
		}

		// Add datapoint to NOTA points if it satisfies minimum nearest neighbor distance requirement
		if (validNOTAPoint) {
			NOTAPoints.push_back(std::move(NOTAPoint));
			++created;
		}
	}

	// Update the state to reflect the total void NOTA points that will be tested on across all folds
	for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
		TEST_RESULTS.voidRandomPoints[pvalCat][idx][0] = created * K_FOLDS;
	}
}

/*
void insertHyperspaceNOTAPoints(const std::vector<std::pair<double, double> >& outerBbox,
	const std::vector<std::pair<double, double> >& innerBbox, NOTACategory NOTALoc, std::mt19937& gen,
	std::vector<ClassMember>& NOTAPoints) {
	size_t total = std::max(MODEL_STATE.datasetSize, MIN_NOTA_POINTS);
	//size_t total = 10;
	size_t dim = outerBbox.size();

	// Create random number generators
	std::vector<std::uniform_real_distribution<double> > featureGenerators(dim);
	for (size_t i = 0; i < dim; ++i) {
		featureGenerators[i] = std::move(std::uniform_real_distribution<double>(outerBbox[i].first, outerBbox[i].second));
	}

	int genAttempts = 0, maxFeatureGenAttempts = 100;

	for (size_t numCreatedPoints = 0; numCreatedPoints < total; ++numCreatedPoints) {
		ClassMember NOTAPoint(std::vector<double>(dim), "NOTA", 0, NOTALoc, -1);

		// Generate features between an inner and outer bounding box
		for (size_t i = 0; i < dim; ++i) {
			for (genAttempts = 0; genAttempts < maxFeatureGenAttempts; ++genAttempts) {
				NOTAPoint.features[i] = featureGenerators[i](gen);
				if (NOTAPoint.features[i] < innerBbox[i].first || NOTAPoint.features[i] > innerBbox[i].second ||
					std::abs(outerBbox[i].first - innerBbox[i].first) < 1e-6) {
					break;
				}
			}

			// If no longer able to create points between the inner and outer bounding box, abort
			if (genAttempts >= maxFeatureGenAttempts) {
				TEST_RESULTS.hyperspaceRandomPoints[NOTALoc][0] = numCreatedPoints * K_FOLDS;
				return;
			}
		}

		NOTAPoints.push_back(std::move(NOTAPoint));
	}

	// Update the state to reflect the total hyperspace NOTA points that will be tested on across all folds
	TEST_RESULTS.hyperspaceRandomPoints[NOTALoc][0] = total * K_FOLDS;
}
*/

void insertHyperspaceNOTAPoints(const std::vector<std::pair<double, double> >& outerBbox,
	const std::vector<std::pair<double, double> >& innerBbox, NOTACategory NOTALoc, std::mt19937& gen,
	std::vector<ClassMember>& NOTAPoints) {
	size_t total = std::max(MODEL_STATE.datasetSize, MIN_NOTA_POINTS);
	//size_t total = 10;
	size_t dim = outerBbox.size();

	// Create random number generators
	std::vector<std::uniform_real_distribution<double> > featureGenerators(dim);
	for (size_t i = 0; i < dim; ++i) {
		featureGenerators[i] = std::move(std::uniform_real_distribution<double>(outerBbox[i].first, outerBbox[i].second));
	}

	int genAttempts = 0, maxFeatureGenAttempts = 100;

	for (size_t numCreatedPoints = 0; numCreatedPoints < total; ++numCreatedPoints) {
		ClassMember NOTAPoint(std::vector<double>(dim), "NOTA", 0, NOTALoc, -1);

		// Generate features between an inner and outer bounding box
		for (size_t i = 0; i < dim; ++i) {
			for (genAttempts = 0; genAttempts < maxFeatureGenAttempts; ++genAttempts) {
				NOTAPoint.features[i] = featureGenerators[i](gen);
				if (NOTAPoint.features[i] < innerBbox[i].first || NOTAPoint.features[i] > innerBbox[i].second ||
					std::abs(outerBbox[i].first - innerBbox[i].first) < 1e-6) {
					break;
				}
			}

			// If no longer able to create points between the inner and outer bounding box, abort
			if (genAttempts >= maxFeatureGenAttempts) {
				for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
					TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTALoc][0] = numCreatedPoints * K_FOLDS;
				}
				return;
			}
		}

		NOTAPoints.push_back(std::move(NOTAPoint));
	}

	// Update the state to reflect the total hyperspace NOTA points that will be tested on across all folds
	for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
		TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTALoc][0] = total * K_FOLDS;
	}
}

// Returns a map from the class name to the largest nearest neighbor distance internal to the class
std::unordered_map<std::string, double> findLargestNNDistancesPerClass(
	const std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
	// Copy datapoints to map from class name to matrix containing feature vectors for members of the class
	std::unordered_map<std::string, std::vector<std::vector<double> > > classMap;
	for (const auto& pair : dataset) {
		classMap[pair.first].reserve(MODEL_STATE.numInstancesPerClass.at(pair.first));
		for (const ClassMember& obj : pair.second) {
			classMap[pair.first].push_back(obj.features);
		}
	}

	// Calculate nearest neighbor distances within each class
	std::unordered_map<std::string, std::vector<double> > nnDistances = computeNearestNeighborDistances(classMap);

	// Find the largest nearest neighbor distance internal to each class
	std::unordered_map<std::string, double> largestNNDistances;
	for (const auto& pair : nnDistances) {
		largestNNDistances[pair.first] = *std::max_element(pair.second.begin(), pair.second.end());
	}

	return largestNNDistances;
}

std::vector<ClassMember> insertRandomPoints(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
	size_t dim = MODEL_STATE.featureMins.size();

	std::unordered_map<std::string, std::vector<std::vector<double> > > matrixDataset;
	for (const auto& pair : dataset) {
		matrixDataset[pair.first].reserve(pair.second.size());
		for (const ClassMember& obj : pair.second) {
			matrixDataset[pair.first].insert(matrixDataset[pair.first].end(), obj.features);
		}
	}

	// Create KD Trees for finding nearest neighbor distances
	std::unordered_map<std::string, KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double>* > kdTrees;
	for (const auto& pair : matrixDataset) {
		kdTrees[pair.first] = new KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double>(dim, pair.second, 10, 0);
	}

	std::random_device rd;
	std::mt19937 gen(rd());

	std::vector<ClassMember> NOTAPoints;

	// Find the largest nearest neighbor distances for each class for different multipliers
	std::unordered_map<std::string, double> largestClassNNDistances = findLargestNNDistancesPerClass(dataset);
	std::vector<std::unordered_map<std::string, double> > minNNDistances(NUM_NN_STEPS);
	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		for (const auto& pair : largestClassNNDistances) {
			minNNDistances[i][pair.first] = pair.second * NNStepMultiplier(i);
		}
	}

	// Create void NOTA points for different multipliers
	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		insertVoidNOTAPoints(MODEL_STATE.featureMins, MODEL_STATE.featureMaxs, i, gen, kdTrees, minNNDistances, NOTAPoints);
	}

	std::vector<double> featureDifs(dim);
	std::vector<std::pair<double, double> > innerBbox(dim);
	std::vector<std::pair<double, double> > outerBbox(dim);

	// Find the outer bounding box for hyperspace NOTA points
	for (size_t i = 0; i < dim; ++i) {
		featureDifs[i] = MODEL_STATE.featureMaxs[i] - MODEL_STATE.featureMins[i];
		outerBbox[i].first = MODEL_STATE.featureMins[i] - (featureDifs[i] * HYPERSPACE_MAX_BBOX_EXTENSION);
		outerBbox[i].second = MODEL_STATE.featureMaxs[i] + (featureDifs[i] * HYPERSPACE_MAX_BBOX_EXTENSION);
	}

	// Create hyperspace NOTA points for different inner bounding boxes
	for (size_t lowerBoundIdx = 0; lowerBoundIdx < HYPERSPACE_LOWER_BOUNDS; ++lowerBoundIdx) {
		for (size_t i = 0; i < dim; ++i) {
			innerBbox[i].first = MODEL_STATE.featureMins[i] - (featureDifs[i] * HYPERSPACE_BBOX_LOWER_BOUNDS[lowerBoundIdx]);
			innerBbox[i].second = MODEL_STATE.featureMaxs[i] + (featureDifs[i] * HYPERSPACE_BBOX_LOWER_BOUNDS[lowerBoundIdx]);
		}

		insertHyperspaceNOTAPoints(outerBbox, innerBbox, static_cast<NOTACategory>(lowerBoundIdx), gen, NOTAPoints);
	}

	for (const auto& kdTree : kdTrees) {
		delete kdTree.second;
	}

	// Beginning of saving NOTA points
	char delim;
	double minNNUnits;
	FILE* fp = fopen(CACHE_PATHS.NOTAPointsFilepath.c_str(), "w");
	fprintf(fp, "category,minNNUnitsFromClass,");
	for (size_t i = 0; i < dim; ++i) {
		delim = (i == dim - 1) ? '\n' : ',';
		fprintf(fp, "col%lu%c", i, delim);
	}
	for (size_t i = 0; i < NOTAPoints.size(); ++i) {
		minNNUnits = (NOTAPoints[i].NOTALocation == NOTACategory::VOID) ?
			NNStepMultiplier(NOTAPoints[i].NNUnitsFromClass) : NOTAPoints[i].NNUnitsFromClass;
		fprintf(fp, "%i,%g,", NOTAPoints[i].NOTALocation, minNNUnits);
		for (size_t j = 0; j < dim; ++j) {
			delim = (j == dim - 1) ? '\n' : ',';
			fprintf(fp, "%g%c", NOTAPoints[i].features[j], delim);
		}
	}
	fclose(fp);

	fp = fopen(CACHE_PATHS.finishedFilepath.c_str(), "w");
	fclose(fp);
	// Ending of saving NOTA points

	return NOTAPoints;
}

std::vector<ClassMember> createNOTAPoints(const std::unordered_map<std::string, std::vector<ClassMember> >& dataset) {
	findFeatureBB(dataset, 0.0);
	return insertRandomPoints(dataset);
}

/*
std::vector<ClassMember> readNOTAPointsFromFile(const std::string& NOTAPointsFilename) {
	std::ifstream NOTAPointsFile(NOTAPointsFilename);
	std::stringstream ss;
	std::string line, field;
	int NOTALoc;
	std::vector<ClassMember> NOTAPoints;

	std::getline(NOTAPointsFile, line);
	while (std::getline(NOTAPointsFile, line)) {
		ss.str("");
		ss.clear();
		ss << line;
		ClassMember NOTAPoint;

		std::getline(ss, field, ',');
		NOTALoc = std::stoi(field);
		NOTAPoint.NOTALocation = static_cast<NOTACategory>(NOTALoc);

		std::getline(ss, field, ',');
		NOTAPoint.NNUnitsFromClass = NNUnitsFromClass(std::stod(field));

		while (std::getline(ss, field, ',')) {
			NOTAPoint.features.emplace_back(std::stod(field));
		}

		NOTAPoint.name = "NOTA";
		NOTAPoint.lineNumber = 0;

		if (NOTAPoint.NOTALocation == NOTACategory::VOID) {
			++TEST_RESULTS.voidRandomPoints[NOTAPoint.NNUnitsFromClass][0];
		}
		else {
			++TEST_RESULTS.hyperspaceRandomPoints[NOTAPoint.NOTALocation][0];
		}

		NOTAPoints.push_back(std::move(NOTAPoint));
	}

	NOTAPointsFile.close();

	for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
		TEST_RESULTS.voidRandomPoints[i][0] *= K_FOLDS;
	}
	for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
		TEST_RESULTS.hyperspaceRandomPoints[static_cast<NOTACategory>(i)][0] *= K_FOLDS;
	}

	return NOTAPoints;
}
*/

std::vector<ClassMember> readNOTAPointsFromFile(const std::string& NOTAPointsFilename) {
	std::ifstream NOTAPointsFile(NOTAPointsFilename);
	std::stringstream ss;
	std::string line, field;
	int NOTALoc;
	std::vector<ClassMember> NOTAPoints;

	std::getline(NOTAPointsFile, line);
	while (std::getline(NOTAPointsFile, line)) {
		ss.str("");
		ss.clear();
		ss << line;
		ClassMember NOTAPoint;

		std::getline(ss, field, ',');
		NOTALoc = std::stoi(field);
		NOTAPoint.NOTALocation = static_cast<NOTACategory>(NOTALoc);

		std::getline(ss, field, ',');
		NOTAPoint.NNUnitsFromClass = NNUnitsFromClass(std::stod(field));

		while (std::getline(ss, field, ',')) {
			NOTAPoint.features.emplace_back(std::stod(field));
		}

		NOTAPoint.name = "NOTA";
		NOTAPoint.lineNumber = 0;

		if (NOTAPoint.NOTALocation == NOTACategory::VOID) {
			for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
				++TEST_RESULTS.voidRandomPoints[pvalCat][NOTAPoint.NNUnitsFromClass][0];
			}
		}
		else {
			for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
				++TEST_RESULTS.hyperspaceRandomPoints[pvalCat][NOTAPoint.NOTALocation][0];
			}
		}

		NOTAPoints.push_back(std::move(NOTAPoint));
	}

	NOTAPointsFile.close();

	for (int pvalCat = 0; pvalCat < PVALUE_NUMERATOR_MAX; ++pvalCat) {
		for (size_t i = 0; i < NUM_NN_STEPS; ++i) {
			TEST_RESULTS.voidRandomPoints[pvalCat][i][0] *= K_FOLDS;
		}
		for (size_t i = 0; i < HYPERSPACE_LOWER_BOUNDS; ++i) {
			TEST_RESULTS.hyperspaceRandomPoints[pvalCat][static_cast<NOTACategory>(i)][0] *= K_FOLDS;
		}
	}

	return NOTAPoints;
}