# include <iostream>
# include "SignalGenerator.h"
#include "Analyzer.h"  // Wieso eig mit "" weil ich die erstellt habe ???? sehr diskriminant digga, was ist fstream


int main() {

    // Variables -------------------
const double Fs = 1000.0; // sampling rate
    const int N = 1000; // Samples
    const double A = 1.4; // Amplitude
    const double f = 50.0; // Frequenz


    SignalGenerator generator(Fs, N, A, f);

}
