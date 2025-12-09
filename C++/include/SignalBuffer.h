//
// Created by richard on 09.12.25.
//

#pragma once
#include <vector>


class  SignalBuffer {
public:

    void addSample(double value);
    const std::vector<double>& getSamples() const;

    private:
    std::vector<double> samples;

};







#ifndef C___SIGNALBUFFER_H
#define C___SIGNALBUFFER_H




#endif //C___SIGNALBUFFER_H