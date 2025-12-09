//
// Created by richard on 09.12.25.
//

#include "../include/SignalBuffer.h"

void SignalBuffer::addSample(double value) {
    samples.push_back(value);
}

const std::vector<double>& SignalBuffer::getSamples() const {
    return samples;
}