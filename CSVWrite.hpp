#ifndef CSVWrite_HPP
#define CSVWrite_HPP

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>

template <typename T> void writeRow(const std::vector<T>& row, std::ofstream& out) {
	std::copy(row.begin(), row.end() - 1, std::ostream_iterator<T>(out, ","));
	out << *(row.end() - 1) << "\n";
}

// Add rows to CSV
template <typename T> void writeToCSV(const std::vector<std::string>& header, const std::vector<std::vector<T> >& rows,
	const std::string& filename) {
	std::ofstream out;
	if (header.size() > 0) {
		out.open(filename);
		// Add a header if this is the first time writing to the CSV
		writeRow<std::string>(header, out);
	}
	else {
		out.open(filename, std::ios::app);
	}
	size_t totalRows = rows.size();
	for (size_t i = 0; i < totalRows; ++i) {
		writeRow<T>(rows[i], out);
	}
	out.close();
}

#endif