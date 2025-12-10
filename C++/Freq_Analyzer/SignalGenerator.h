//
// Created by richard on 10.12.25.
//

#ifndef FREQ_ANALYZER_SIGNALGENERATOR_H
#define FREQ_ANALYZER_SIGNALGENERATOR_H
//-_----------------------------- brAH ES GEHT LOS
// HEADER FILE (BALD KRIEGE ICH AUCH EINE KUGEL IN MEIN HEAD)

#pragma once  // Was zum dick heißt das pragma ? Dota 2 Klasse ?
#include <vector>
#include <cmath>  // Wie kann ich eig nochmal cmath einsehen brah ?

class SignalGenerator {

private:

    double Fs; // Samples per Second
    double N; // Samples Anzahl
    double A; // Amplitude
    double f; // frequency

    // f ist für Frequenz A für Amplitude N ist für Samples  in mathe und Fs ist für die Samplrate pro sekunde
    // Die ist im protected drinne denn nur die Vars nutzt die klasse für die berechnung in den späteren methoden

public:
SignalGenerator(double Fs, double N, double A, double f) // Und hier werden die Attribute der Objekte reingeschmissen ?
    : Fs(Fs), N(N), A(A), f(f) {} // Ich checke iwie die zeile hier garnicht ist es der output der F oder wie ?
    // Denn der berreich da drinne ist ja da leer ?????? ich weiß halt header und so und iwie aber er kann doch gefüllt werden

std::vector<double> generateSine() {
    std::vector<double> samples;  // Hier wird doch ein Vec Erschaffen oder ?
    samples.reserve(N); // Digga hääää sagst du schon da dass er die anzahl n vorhalten soll ?

for (int n = 0; n < N; n++) { // ich checke die bedingung nicht so ganz tbh... also wenn die gleitzahl n größer als die Anzahl der Samples ist ?
    // Itteriere ich einf sozusagen die anzahl an Samples
    double x = A * sin(2 * M_PI * n / Fs);
        samples.push_back(x);
}
return samples;
}







protected:


};


#endif //FREQ_ANALYZER_SIGNALGENERATOR_H