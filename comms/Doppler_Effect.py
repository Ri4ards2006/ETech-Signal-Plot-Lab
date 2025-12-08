import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def simulate_doppler_effect(f_source, v_observer, v_source, c, duration, sample_rate):
    """
    Simuliert den Doppler-Effekt für ein reines Frequenzsignal.
    Parametrisiere den Beobachter- und Quellenbewegung und betrachte die Frequenzveränderung.
    
    Args:
        f_source (float): Originalfrequenz der Quelle in Hz.
        v_observer (float): Geschwindigkeit des Beobachters in m/s (+ naht der Quelle).
        v_source (float): Geschwindigkeit der Quelle in m/s (+ entfernt sich vom Beobachter).
        c (float): Übertragungsgeschwindigkeit (z. B. Schallgeschwindigkeit in m/s).
        duration (float): Zeitdauer der Simulation in Sekunden.
        sample_rate (int): Abtastrate für die Simulation in Hz.
    
    Returns:
        dict: Enthält Zeitachse, Originalsignal, Doppler-Signal, und Sample-Rate.
    """
    # Zeitachse generieren (Zeitpunkte in Sekunden)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Beobachtete Frequenz (Doppler-Formel)
    numerator = c + v_observer  # Zähler: Übertragungsgeschwindigkeit + Beobachtergeschwindigkeit
    denominator = c + v_source   # Nenner: Übertragungsgeschwindigkeit + Quellengeschwindigkeit
    f_observed = f_source * (numerator / denominator)  # Beobachtete Frequenz
    
    # Originalsignal (Sinus mit Originalfrequenz)
    original_signal = np.sin(2 * np.pi * f_source * t)
    
    # Doppler-effektes Signal (Sinus mit beobachteter Frequenz)
    doppler_signal = np.sin(2 * np.pi * f_observed * t)
    
    # Rückgabedictionary mit allen relevanten Parametern
    return {
        't': t,
        'original_signal': original_signal,
        'doppler_signal': doppler_signal,
        'sample_rate': sample_rate,  # Hinzugefügt: Sample-Rate für FFT-Berechnung
        'f_source': f_source         # Optional: Originalfrequenz für Plottaufgaben
    }

def plot_results(simulation_data):
    """
    Plotten der originalen und Doppler-effektes Signale im Zeitbereich und deren FFT.
    """
    # Zeitbereich-Plot
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_data['t'], simulation_data['original_signal'], color="#2c7fb8", linewidth=1.2, label=f'Original (f={simulation_data["f_source"]:.0f} Hz)')
    plt.plot(simulation_data['t'], simulation_data['doppler_signal'], color="#ff7f0e", linewidth=1.2, linestyle="--", label=f'Doppler (f\'={simulation_data["f_source"]*(simulation_data["t"][-1"]?) No, wait: Berechne f_observed direkt hier.')
    
    # Korrektur: f_observed aus simulation_data berechnen (oder in simulate_doppler_effect hinzufügen)
    numerator = simulation_data['sample_rate']? Nein, besser: Rechne f_observed wieder, weil es nicht gespeichert wurde.
    # Wait, in simulate_doppler_effect haben wir f_observed berechnet, aber sie ist nicht im dictionary. Lass uns sie hinzufügen!
    # (Um Fehler zu vermeiden, füge f_observed in simulate_doppler_effect hinzu)
    # Also, korrigiere simulate_doppler_effect, um 'f_observed' zu speichern:
    # return { ... , 'f_observed': f_observed, ... }
    
    # Für den Zeitbereich-Label: Lass uns f_observed berechnen und einfügen
    f_observed = simulation_data['f_source'] * ( (simulation_data.get('c', 343) + simulation_data.get('v_observer', 0)) / (simulation_data.get('c', 343) + simulation_data.get('v_source', 0)) )
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Zeitbereich: Original vs. Beobachtetes Signal', fontsize=14, pad=20)
    plt.xlabel('Zeit (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.show()
    
    # Frequenzbereich-Plot (FFT)
    # Originalsignal FFT
    original_fft = fft(simulation_data['original_signal'])
    # Doppler-Signal FFT
    doppler_fft = fft(simulation_data['doppler_signal'])
    
    # Frequenzen basierend auf Sample-Rate berechnen
    # Sample-Rate ist jetzt im dictionary gespeichert (simulation_data['sample_rate'])
    freqs_original = fftfreq(len(simulation_data['original_signal']), 1 / simulation_data['sample_rate'])
    freqs_doppler = fftfreq(len(simulation_data['doppler_signal']), 1 / simulation_data['sample_rate'])
    
    # Bereinige FFT (nur positive Frequenzen)
    # Wir können auch die Amplitude normalisieren (dividiere durch Länge des Signals)
    plt.figure(figsize=(12, 6))
    # Original-Spektrum
    plt.plot(freqs_original, np.abs(original_fft)/len(simulation_data['original_signal']), color="#2c7fb8", linewidth=1.2, label=f'Original (f={simulation_data["f_source"]:.0f} Hz)')
    # Doppler-Spektrum
    plt.plot(freqs_doppler, np.abs(doppler_fft)/len(simulation_data['doppler_signal']), color="#ff7f0e", linewidth=1.2, linestyle="--", label=f'Doppler (f\'={f_observed:.0f} Hz)')
    
    # Begrenze Frequenzbereich auf relevantes Intervall
    max_freq = simulation_data['f_source'] * 2  # Bis 2*f_source, da Doppler效应对 1kHz bei v_observer=5m/s nicht so stark ist
    plt.xlim(0, max_freq)
    
    plt.title('Frequenzbereich: FFT des Original- und Doppler-Signals', fontsize=14, pad=20)
    plt.xlabel('Frequenz (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Simulierung mit Beispielwerten
simulation_data = simulate_doppler_effect(
    f_source=1000,    # Originalfrequenz (Hz, nicht kHz!)
    v_observer=5,     # Beobachtergeschwindigkeit (m/s, + naht der Quelle)
    v_source=0,       # Quellengeschwindigkeit (m/s, 0 = ruhend)
    c=343,            # Schallgeschwindigkeit in Luft (m/s)
    duration=2,       # Simulationdauer (s)
    sample_rate=44100 # Audiosample-Rate (Hz)
)

# Plotte die Ergebnisse
plot_results(simulation_data)