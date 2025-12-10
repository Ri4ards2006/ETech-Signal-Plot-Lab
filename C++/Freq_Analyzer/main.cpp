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
// ab hier gehts mit der Berechnung der 0 Crossing Frequenz
    int zeroCrossings = 0;

    for (int i = 1; i < samples.size(); i++) {
        if ((samples[i-1] > 0 && samples[i] <= 0) || (samples[i-1] < 0 && samples[i] >= 0)) {
            zeroCrossings++;
        }
    }

    double estimatedFrequency = (zeroCrossings / 2.0) * (Fs / samples.size());
    std::cout << "Frequenz: " << estimatedFrequency << " Hz\n";


    return 0;
}

