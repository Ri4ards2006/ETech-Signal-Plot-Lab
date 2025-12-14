//
// Created by richard on 11.12.25.
//

#ifndef FREQ_ANALYZER_TIMER_H
#define FREQ_ANALYZER_TIMER_H
#include <cstdint>


class Timer {
public:
    void init(uint16_t interval_ms) {
        // konfiguration des Timers
    }

    void attachInterrupt(void (*callback)()) {
        // Funktion ausführen bei Timer-Interrupt
    }

};


#endif //FREQ_ANALYZER_TIMER_H