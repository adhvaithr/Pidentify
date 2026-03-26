#include <fstream>
#include <string>

#include "classMember.h"

class DatasetReader {
	public:
		DatasetReader(const std::string& filename);
		ClassMember getNextPoint();
		~DatasetReader();

	private:
		std::ifstream file;
		size_t numFeatures = 0;
		size_t lineNumber = 2;
};
