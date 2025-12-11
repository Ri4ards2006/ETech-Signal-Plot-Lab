// Die Klasse ist für den Sammel von samples da in einem Ring Buffer
// Created by richard on 11.12.25.
//

#ifndef FREQ_ANALYZER_ADCBUFFER_H
#define FREQ_ANALYZER_ADCBUFFER_H
#include <cstdint>


class ADCBuffer {
private:
    static const int SIZE = 128; // Warum isst es so groß Geschrieben wtf ???? und warum statisch
    uint16_t buffer[SIZE]; // Was ist der datentyp und ist es jz ein array oder wtf ?????
    int index =0;

public:
    void addSample(uint16_t val) /* Was ist dieses val jz ? ist es der name davon es ist so fking komisch */ {


    }





};

#endif //FREQ_ANALYZER_ADCBUFFER_H