//
// Created by richard on 10.12.25.
//
#ifndef FREQ_ANALYZER_ANALYZER_H
#define FREQ_ANALYZER_ANALYZER_H

// Was zum f sind eig die 2 Zeilen da drüber ????

#include <cmath>
#include <vector>
#pragma once; // Was zum fick soll denn das ???
class Analyzer {

public:
    /* Also ich deklariere hier eine Statische Methode mit dem output double die die 3 Parametor hat nh ?
     * Also den vector von Doubles namens Samples und die Samples pro sekunde oder ? und die wird zu f umgerechnet nh?
     */
    static double zeroCrossingFreq(const std::vector<double>& samples, double Fs) {
int zeroCrossing = 0;  // Hier wird die Int Var namens zeroCrossing initiert
        for (size_t i = 1; i < samples.size(); ++i ) { // Ist size_t ei eine var oder was genau ich checke es nicht ?
            // Denn dann wird sie iwie als eine fucking methode verwendet es verwirrt mich total tbh
if ((samples[i-1] < 0 && samples[i] >= 0 ) || (samples[i-1] > 0 && samples[i] <= 0))
// und hier kommt der fette und ficke batz von wenn in dem vec namens samples die bedingung das wenn [i-1]  kleiner als 0 ist
// und gleiczeitig der jetzige sample denn du hast größer oder gleich 0 ist eintritt dann weißt du das er über die x achse
// Drüber gejumped ist damit du halt später das für die berechnung nutze nh ?????  und dann halt das selbe in oder aber halt nur
// von + in - und das davor war von + in -
    {
    // Dann wird der wert der Variable zero crossing erhöht heißt also wie oft du  die 0 linie überschritte hast
    zeroCrossing ++ ; // Brah sagt denn wie oft deine Sinus kurve beef gemacht hast
            }
        }
        return (zeroCrossing * 2.0) * (Fs / samples.size());
        // Was soll denn die rechnung wieso mach ich es mall 2 ?? wegen der periode oder wie ???  und dann
        // noch * die Fs durch sample size methode eyyy so strange achsooooo ist es f für  omega ??? aber iwie ohne pi ?
    }

// ----------------------   Jetzt Definite Fourier Transformation ------------------------------

    // Wieso brauch mann für die Prozedur eig nur den einen Vector Parameter wo ist denn f  A oder iwie was anderes ?
    static std::vector<double> computeDFT(const std::vector<double>& samples) {
        int N = samples.size();

        std::vector<double> mag(N/2); // Was soll denn jetzt das schon wieder für was steht mag ???? und denn noch /2

        for (int k = 0; k < N/2; k ++) {
            double re = 0.0;
            double im = 0.0; // Wieso wusste die auto ai was zum fick ist mit re und im gemeint ?????
            // Sag nicht das du hier in Mengenberreich "C" reingehst ...
            // Ich dachte das definite das auf chillig auf jeden macht und nix mit
            // UHHHH Schau mich an ich bin besonders und "CoMpLeX" kys

            for (int n = 0; n < N; n++) {// ist es damit er alle Samples durchläuft ?
            double angle = 2 * M_PI * n / N; // Wozu das M eig ???? also ist n/N  = f ???ß
                re += samples[n] * cos(angle);
                im -= samples[n] * sin(angle);
            }
            mag[k] = sqrt(re * re + im* im); // Pythogaros nh ? für den kram aber was zum fick ist mag ????
            // ich checke nicht alles und viel aber wie ist mein verständnis so eig ??
        }
    }

};


#endif //FREQ_ANALYZER_ANALYZER_H  // Was soll den das schon wieder hier eyyyy ????