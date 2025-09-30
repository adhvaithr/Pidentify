#ifndef CLASSMEMBER_H
#define CLASSMEMBER_H

#include <vector>
#include <string>

struct ClassMember {
    std::vector<double> features;
    std::string name;
    size_t lineNumber;
    bool NOTA;

    ClassMember() : features(std::vector<double>(0)), name(""), lineNumber(0), NOTA(false) {}
    ClassMember(const std::vector<double>& featureDatapoints, const std::string& className,
        size_t line, bool isNOTA = false) : features(featureDatapoints), name(className),
        lineNumber(line), NOTA(isNOTA) {};
};

#endif