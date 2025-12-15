//
// Created by richard on 15.12.25.
//

#ifndef FREQ_ANALYZER_RINGBUFFER_H
#define FREQ_ANALYZER_RINGBUFFER_H

#include <cstdint> // Was ist das für eine Lib Genau ????

template <typename T,  uint16_t SIZE> // Initiere einen Template für all den
// Komischen kram.. ,  also die jeweilige größe ist ja 16 Bit aber was dast ???
// Steht das mit der  lib in verbindung ?

// Achso typ und feste größe zur Compile-Zeit
// "SIZE"  Quasi ein Const für den Kram
class RingBuffer {

    T Buffer[SIZE]; // Eine Fest Größe für die klasse die hier Privat
    // Deklariert wird
    volatile uint16_t head;// Die Schreibposition also der Kopf
    volatile uint16_t tail; // Leseposition
    volatile bool full; // Heißt das nichts optimiert und verbessert wird.
};  // Volatile ISR Schreibt
// Main Loop Liest also wird nix vom compiler optimiert

// Constructor :
 public:
RingBuffer() : head(0), tail(0), full(false) {}
// Also jz würde das Objekt halt mit einem Guten Heap management Erzeugt werden


#endif //FREQ_ANALYZER_RINGBUFFER_H