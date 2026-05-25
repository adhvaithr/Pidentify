#include <sstream>

#include "readDataset.h"

DatasetReader::DatasetReader(const std::string& filename) {
    file.open(filename);
    std::string line;

    // Read the header
    std::getline(file, line);
    std::stringstream header(line);
    std::string colName;
    std::getline(header, colName, ',');
    // Count the number of numerical features
    while (std::getline(header, colName, ',')) {
        if (colName.compare(0, 6, "nonNum") == 0) {
            break;
        }
        numFeatures += 1;
    }
}

ClassMember DatasetReader::getNextPoint() {
    std::string line;
    ClassMember obj;

    // Only add the numerical features to the ClassMember vector
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string feature;
        std::getline(ss, obj.name, ',');
        for (size_t i = 0; i < numFeatures; ++i) {
            std::getline(ss, feature, ',');
            obj.features.push_back(std::stod(feature));
        }
        obj.lineNumber = lineNumber++;
    }

    return obj;
}

DatasetReader::~DatasetReader() {
    file.close();
}
