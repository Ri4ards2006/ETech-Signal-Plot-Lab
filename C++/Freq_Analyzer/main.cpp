#include <iostream>   // Für std::cout
#include <vector>     // Für std::vector
#include <cmath>      // Für sin(), M_PI

int main() {
    // =========================
    // 1️⃣ Konstanten definieren
    // =========================
    const double Fs = 1000.0; // Abtastrate: 1000 Samples pro Sekunde
    const double f = 50.0;    // Signal-Frequenz 50 Hz
    const double A = 1.0;     // Amplitude
    const int N = 1000;       // Anzahl Samples

    // =========================
    // 2️⃣ Signal in einen Vector speichern
    // =========================
    std::vector<double> samples;  // Vector für alle Samples
    samples.reserve(N);           // Speicherplatz reservieren (optional, effizienter)

    for (int n = 0; n < N; n++) {
        double x = A * sin(2.0 * M_PI * f * n / Fs); // Sinus-Signal erzeugen
        samples.push_back(x);                         // Sample in den Vector packen
    }

    // Optional: erste 20 Samples ausgeben
    std::cout << "Erste 20 Samples:\n";
    for (int i = 0; i < 20; i++) {
        std::cout << samples[i] << "\n";
    }

    // =========================
    // 3️⃣ Zero-Crossing Zählen
    // =========================
    int zeroCrossings = 0; // Zähler für Nulldurchgänge

    // Wir starten bei i = 1, um samples[i-1] benutzen zu können
    for (int i = 1; i < samples.size(); i++) {
        // Prüfen, ob das Signal die Nulllinie kreuzt
        if ((samples[i-1] > 0 && samples[i] <= 0) ||
            (samples[i-1] < 0 && samples[i] >= 0)) {
            zeroCrossings++;
            }
    }

    // =========================
    // 4️⃣ Frequenz berechnen
    // =========================
    double estimatedFrequency = (zeroCrossings / 2.0) * (Fs / samples.size());

    std::cout << "Geschätzte Frequenz: " << estimatedFrequency << " Hz\n";

    return 0; // Programm erfolgreich beendet
}
