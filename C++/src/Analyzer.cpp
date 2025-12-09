//
// Created by richard on 09.12.25.
//

#include "../include/Analyzer.h"
#include <limits>

double Analyzer::mean(const std :: vector<double>& data) {
    double sum = 0.0;

    for (double value : data) {
        sum += value;
    }
    return data.empty() ? 0.0 : sum / data.size();
}

double Analyzer::min(const std::vector<double>& data) {
    if (data.empty()) return 0.0;

    double m = std::numeric_limits<double>::infinity();
    for (double v : data) {
        if (v < m) m = v;
    }
    return m;
}

double Analyzer::max(const std::vector<double>& data) {
    if (data.empty()) return 0.0;

    double m = -std::numeric_limits<double>::infinity();
    for (double v : data) {
        if (v > m) m = v;
    }
    return m;
}