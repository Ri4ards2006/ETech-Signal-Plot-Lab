#pragma once
#include <vector>

class Analyzer {
public:
    static double mean(const std::vector<double>& data);
    static double min(const std::vector<double>& data);
    static double max(const std::vector<double>& data);
};
