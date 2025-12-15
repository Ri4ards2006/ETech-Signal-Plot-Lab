# include <iostream>
#include "signal/SignalGenerator.h"
#include "dsp/Analyzer.h"


int main() {

    // Variables -------------------
const double Fs = 1000.0; // sampling rate
    const int N = 1000; // Samples
    const double A = 1.4; // Amplitude
    const double f = 50.0; // Frequenz



    SignalGenerator generator(Fs, N, A, f); //  Initierung des Signal Generator objekts oder ????ßßßß


    // Wählen zwischen Sinus oder Square Signalen


    auto samples = generator.generateSine();
    // Auto Samples wie funktioniert das eig digga ???



    double f_est = Analyzer :: zeroCrossingFreq(samples,Fs); // Initierung der f est variable und Zero Cross methode oder iwie so hiilfe
    std::cout << "Estimated Frequency (Zero Crossing ):"<< f_est ;  // Was ist dieses dieses <<<<<< eig digga

auto spectrum = Analyzer :: computeDFT(samples); // Nur der vec aufruf der deklarierten methode computeDFT vom header  vom Analyzer
 for (size_t k = 0; k < 20; k++) { // Wie viele und wie genau oder ???
double freq = k* Fs/ N ; // für Omega oder ?
     std::cout << "f=" << freq << "Hz, Magnitude" << spectrum[k] << "\n"; // Digga warum << statt + ?????????ßß
 }
    return 0; // ist gut nh wie die chance wie ka kys
}

