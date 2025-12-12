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

buffer[index++] =  val; // Inwiefern ?? ist es denn jz das problem das er den werd um einen erhöht also die variable oder wie
        // Und warum immer []  es verwirrt mich
        if (index >= SIZE) index = 0;  // Warum darf mann das ?
    }
    uint16_t geSample(int i ){ return buffer[i];}
};

#endif //FREQ_ANALYZER_ADCBUFFER_H

// Mir wurde noch explizit gesagt das es da keine Heap Allocation und keine eigene Header ???