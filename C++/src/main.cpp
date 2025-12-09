#include <iostream>
#include <cmath>
#include <cstdlib>
#include "../include/SignalBuffer.h"
#include "../include/Analyzer.h"


int main() {
    SignalBuffer buffer;

    // Generiere 100 Samples eines verrauschten Sinus
    for (int i = 0; i < 100; i++) {
        double t = i * 0.1;
        double noise = ((rand() % 200) - 100) / 500.0; // ±0.2
        double value = std::sin(t) + noise;
        buffer.addSample(value);
    }

    const auto& samples = buffer.getSamples();

    std::cout << "Mean: " << Analyzer::mean(samples) << "\n";
    std::cout << "Min:  " << Analyzer::min(samples) << "\n";
    std::cout << "Max:  " << Analyzer::max(samples) << "\n";

    return 0;
}
