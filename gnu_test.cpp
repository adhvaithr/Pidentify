#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

// Function to generate data for a normal distribution
std::vector<std::pair<double, double>> generateNormalDistribution(double mean, double stddev, double min, double max, double step) {
    std::vector<std::pair<double, double>> data;
    for (double x = min; x <= max; x += step) {
        double y = exp(-0.5 * pow((x - mean) / stddev, 2)) / (stddev * sqrt(2 * M_PI));
        data.push_back(std::make_pair(x, y));
    }
    return data;
}

int main() {
    // Open a pipe to gnuplot
    FILE *pipe = popen("gnuplot -persist", "w");
    if (!pipe) {
        std::cerr << "Unable to open pipe to Gnuplot" << std::endl;
        return 1;
    }

    // Generate normal distribution data
    double mean = 0.0;
    double stddev = 1.0;
    double min = -5.0;
    double max = 5.0;
    double step = 0.1;
    std::vector<std::pair<double, double>> normalData = generateNormalDistribution(mean, stddev, min, max, step);

    // Plot the normal distribution
    fprintf(pipe, "set title 'Normal Distribution'\n");
    fprintf(pipe, "set xlabel 'x'\n");
    fprintf(pipe, "set ylabel 'Probability Density'\n");
    fprintf(pipe, "plot '-' with lines title 'Normal Distribution'\n");
    for (const auto &point : normalData) {
        fprintf(pipe, "%f %f\n", point.first, point.second);
    }
    fprintf(pipe, "e\n");
    fflush(pipe); // flush the pipe

    std::cout << "Press enter to exit" << std::endl;
    std::cin.get();

    // Close the pipe
    pclose(pipe);

    return 0;
}