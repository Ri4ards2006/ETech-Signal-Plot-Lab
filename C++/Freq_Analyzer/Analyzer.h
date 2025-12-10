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


};


#endif //FREQ_ANALYZER_ANALYZER_H  // Was soll den das schon wieder hier eyyyy ????