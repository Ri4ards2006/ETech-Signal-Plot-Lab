// Die Klasse ist für den Sammel von samples da in einem Ring Buffer
// Created by richard on 11.12.25.
//

#ifndef FREQ_ANALYZER_ADCBUFFER_H
#define FREQ_ANALYZER_ADCBUFFER_H


class ADCBuffer {
private:
    static const int SIZE = 128; // Warum isst es so groß Geschrieben wtf ???? und warum statisch
    uint16_t buffer[SiZE]; // Was ist der datentyp und ist es jz ein array oder wtf ?????
     int index =0
};


#endif //FREQ_ANALYZER_ADCBUFFER_H