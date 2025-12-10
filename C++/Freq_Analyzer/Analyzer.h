//
// Created by richard on 10.12.25.
//
#ifndef FREQ_ANALYZER_ANALYZER_H
#define FREQ_ANALYZER_ANALYZER_H

// Was zum f sind eig die 2 Zeilen da drüber ????

#include <cmath>
#include <vector>
#pragma once;
class Analyzer {

public:
    /* Also ich deklariere hier eine Statische Methode mit dem output double die die 3 Parametor hat nh ?
     * Also den vector von Doubles namens Samples und die Samples pro sekunde oder ? und die wird zu f umgerechnet nh?
     */
    static double zeroCrossingFreq(const std::vector<double>& samples, double Fs) {
int zeroCrossing = 0;  // Hier wird die Int Var namens zeroCrossing initiert
        for (size_t i = 1; i < samples.size(); ++i ) { // Ist size_t ei eine var oder was genau ich checke es nicht ?
            // Denn dann wird sie
if ((samples[i-1] < 0 && samples[i] >= 0 ) || (samples[i-1] > 0 && samples[i] <= 0)) {



}

        }
    }



};


#endif //FREQ_ANALYZER_ANALYZER_H