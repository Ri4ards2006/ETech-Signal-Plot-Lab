import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def simulate_doppler(
    f_source: float = 1000,    # Originalfrequenz der Quelle (Hz)
    v_observer: float = 5,    # Beobachtergeschwindigkeit (m/s, + = auf Quelle zu)
    v_source: float = 0,      # Quellgeschwindigkeit (m/s, + = weg von Beobachter)
    c: float = 343,           # Schallgeschwindigkeit (m/s)
    duration: float = 2,      # Signal-Dauer (Sekunden)
    sample_rate: int = 44100, # Abtastrate (Hz, für Audiodaten realistisch)
    plot_title: str = "Doppler-Effekt Simulation: Original vs. Beobachtes Signal"
):
    # 1. Zeitachse generieren
    t = np.linspace(0, duration, int(sample_rate * duration))  # Zeit in Sekunden
    
    # 2. Originalsignal (Quelle)
    original_signal = np.sin(2 * np.pi * f_source * t)  # Amplitude normiert auf 1
    
    # 3. Doppler-Parameter berechnen
    # Beobachter naht der Quelle: v_observer + → f' ↑
    # Quelle naht dem Beobachter: v_source - → f' ↑
    numerator = c + v_observer  # Zähler: c + v_observer (Beobachterbewegung)
    denominator = c + v_source  # Nenner: c + v_source (Quellenbewegung)
    f_observed = f_source * (numerator / denominator)  # Beobachtete Frequenz
    
    # 4. Doppler-beschertes Signal (Beobachter erhält)
    # Annahme: Quelle bewegt sich gleichmäßig; Signal wird verzerrt basierend auf f_observed
    # (Hier vereinfacht: Signal wird direkt mit f_observed generiert; realer Fall könnte Verzögerungen beinhalten)
    doppler_signal = np.sin(2 * np.pi * f_observed * t)
    
    # 5. Frequenzspektra berechnen (FFT)
    def compute_spectrum(signal, sample_rate):
        n = len(signal)
        y_f = fft(signal)
        y_f_mag = np.abs(y_f) / n  # Amplitude normalisieren
        freqs = fftfreq(n, 1 / sample_rate)
        return freqs[:n//2], y_f_mag[:n//2]  # Nur positive Frequenzen
    
    freqs_orig, spectrum_orig = compute_spectrum(original_signal, sample_rate)
    freqs_doppler, spectrum_doppler = compute_spectrum(doppler_signal, sample_rate)
    
    # 6. Plot-Einstellungen (cool & schön!)
    plt.style.use("seaborn-darkgrid")  # Elegantes Design
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    fig.suptitle(plot_title, fontsize=16, y=0.95)
    
    # Zeitbereich-Plot (oben)
    ax_time.plot(t, original_signal, color="#2c7fb8", linewidth=1.2, label=f"Original (f={f_source:.0f} Hz)")
    ax_time.plot(t, doppler_signal, color="#ff7f0e", linewidth=1.2, linestyle="--", label=f"Doppler (f'={f_observed:.0f} Hz)")
    ax_time.set_title("Zeitbereich: Original vs. Beobachtes Signal", fontsize=14)
    ax_time.set_xlabel("Zeit (s)", fontsize=12)
    ax_time.set_ylabel("Amplitude", fontsize=12)
    ax_time.legend()
    ax_time.grid(True, alpha=0.5)
    
    # Frequenzbereich-Plot (unten)
    ax_freq.plot(freqs_orig, spectrum_orig, color="#2c7fb8", linewidth=1.2, label="Originalspektrum")
    ax_freq.plot(freqs_doppler, spectrum_doppler, color="#ff7f0e", linewidth=1.2, linestyle="--", label="Doppler-Spektrum")
    ax_freq.set_title("Frequenzbereich: FFT des Original- und Doppler-Signals", fontsize=14)
    ax_freq.set_xlabel("Frequenz (Hz)", fontsize=12)
    ax_freq.set_ylabel("Amplitude", fontsize=12)
    ax_freq.legend()
    ax_freq.grid(True, alpha=0.5)
    ax_freq.set_xlim(0, 2*f_source)  # Begrenze Frequenzbereich für Klarheit
    
    plt.tight_layout()  # Optimiere Layout
    plt.show()

# Beispielaufruf: Beobachter bewegt sich auf 1 kHz-Quelle zu (v_observer=5 m/s, Quelle ruhend)
simulate_doppler(
    f_source=1000,
    v_observer=5,  # Beobachter naht der Quelle (f' steigt)
    v_source=0,
    c=343,
    duration=2,
    sample_rate=44100,
    plot_title="Doppler-Effekt: Beobachter auf Quelle zu (v=5 m/s)"
)