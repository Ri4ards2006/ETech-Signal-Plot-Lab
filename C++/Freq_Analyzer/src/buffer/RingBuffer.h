//
// Created by richard on 15.12.25.
//

#ifndef FREQ_ANALYZER_RINGBUFFER_H
#define FREQ_ANALYZER_RINGBUFFER_H

#include <cstdint> // Die lib definiert die genaue größe von Int Typen
// Es ist wichtig damit die Fest geräteübergreifend sind..
// uint16_t ist klar und deterministisch !!!
template <typename T,  uint16_t SIZE> //Es ist ein KompilierZeit Konstant Wert
// von dem typ uint16_t

class RingBuffer {

    T Buffer[SIZE]; // An Real C Array
    // Deklariert wird
    volatile uint16_t head;// Die Schreibposition also der Kopf
    volatile uint16_t tail; // Leseposition

    volatile bool full; // Heißt das es außerhalb des codes Ändern kann
    // Also darf der Compiler es nicht cachen und muss es jedes mall neu  Lesen
public:
    RingBuffer() : head(0), tail(0), full(false) {}
    // Also jz würde das Objekt halt mit einem Guten Heap management Erzeugt werden


    bool push(T& value) // effiziente benutzung von C++ Space
    {
        if (full) {
            return false;
        }

        buffer[head] = value;
        head = (head + 1) % SIZE;

        if (head == tail) {
            full = true;
        }
        return true;
    }
};  // Volatile ISR Schreibt
// Main Loop Liest also wird nix vom compiler optimiert

// Constructor :

#endif //FREQ_ANALYZER_RINGBUFFER_H