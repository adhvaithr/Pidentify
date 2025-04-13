#ifndef CLASSMEMBER_H
#define CLASSMEMBER_H

#include <vector>
#include <string>

struct ClassMember {
    std::vector<double> features;
    std::string name;
    std::size_t lineNumber;
};

#endif