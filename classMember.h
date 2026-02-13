#ifndef CLASSMEMBER_H
#define CLASSMEMBER_H

#include <vector>
#include <string>

enum NOTACategory {HYPERSPACE0, HYPERSPACE10, HYPERSPACE20, HYPERSPACE30, HYPERSPACE40, HYPERSPACE50,
                   HYPERSPACE60, HYPERSPACE70, HYPERSPACE80, HYPERSPACE90, HYPERSPACE100, VOID};

struct ClassMember {
    std::vector<double> features;
    std::string name;
    size_t lineNumber;
    NOTACategory NOTALocation;
    int NNUnitsFromClass;

    ClassMember() : features(std::vector<double>(0)), name(""), lineNumber(0) {}
    ClassMember(const std::vector<double>& featureDatapoints, const std::string& className,
        size_t line, NOTACategory loc, double minNNUnits) : features(featureDatapoints), name(className),
        lineNumber(line), NOTALocation(loc), NNUnitsFromClass(minNNUnits) {};
};

#endif