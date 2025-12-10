#include <iostream>
#include <vector>
#include <cmath>

int main() {
    const double Fs = 1000.0; // Sampling-Rate 1000 Hz
    const double f = 50.0;    // Signal-Frequenz 50 Hz
    const double A = 1.0;     // Amplitude
    const int N = 1000;       // Anzahl Samples

    std::vector<double> samples;
    samples.reserve(N);

    for (int n = 0; n < N; n++) {
        double x = A * sin(2.0 * M_PI * f * n / Fs);
        samples.push_back(x);
    }

    for (int i = 0; i < 20; i++) {
        std::cout << samples[i] << "\n";
    }

    return 0;
}
