# Simulierung und Plotting Doppler-Effekt
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
        dict: Enthält Zeitachse, Originalsignal, Doppler-Signal, Sample-Rate und beobachtete Frequenz.
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
        'sample_rate': sample_rate,
        'f_source': f_source,
        'f_observed': f_observed  # Hinzugefügt: Beobachtete Frequenz für Labels
    }

def plot_results(simulation_data):
    """
    Plotten der originalen und Doppler-effektes Signale im Zeitbereich und deren FFT.
    """
    # Zeitbereich-Plot
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_data['t'], simulation_data['original_signal'], color="#2c7fb8", linewidth=1.2, 
             label=f'Original (f=${simulation_data["f_source"]:.0f}$ Hz)')
    plt.plot(simulation_data['t'], simulation_data['doppler_signal'], color="#ff7f0e", linewidth=1.2, linestyle="--", 
             label=f'Doppler (f\'=${simulation_data["f_observed"]:.0f}$ Hz)')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Zeitbereich: Original vs. Beobachtetes Signal', fontsize=14, pad=20)
    plt.xlabel('Zeit (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.show()
    
    # Frequenzbereich-Plot (FFT)
    original_fft = fft(simulation_data['original_signal'])
    doppler_fft = fft(simulation_data['doppler_signal'])
    
    # Berechne Frequenzachsen mit FFT-Funktion
    freqs_original = fftfreq(len(simulation_data['original_signal']), 1 / simulation_data['sample_rate'])
    freqs_doppler = fftfreq(len(simulation_data['doppler_signal']), 1 / simulation_data['sample_rate'])
    
    plt.figure(figsize=(12, 6))
    # Plot Original-Spektrum
    plt.plot(freqs_original, np.abs(original_fft)/len(simulation_data['original_signal']), 
             color="#2c7fb8", linewidth=1.2, label=f'Original (f=${simulation_data["f_source"]:.0f}$ Hz)')
    # Plot Doppler-Spektrum
    plt.plot(freqs_doppler, np.abs(doppler_fft)/len(simulation_data['doppler_signal']), 
             color="#ff7f0e", linewidth=1.2, linestyle="--", label=f'Doppler (f\'=${simulation_data["f_observed"]:.0f}$ Hz)')
    
    plt.xlim(0, simulation_data['f_source'] * 2)
    plt.title('Frequenzbereich: FFT des Original- und Doppler-Signals', fontsize=14, pad=20)
    plt.xlabel('Frequenz (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Simulierung mit Beispielwerten
simulation_data = simulate_doppler_effect(
    f_source=1000,    # Originalfrequenz (Hz)
    v_observer=5,     # Beobachter bewegt sich auf die Quelle zu (+)
    v_source=0,       # Quelle ist ruhend (0)
    c=343,            # Schallgeschwindigkeit (m/s)
    duration=2,       # Simulationsdauer (s)
    sample_rate=44100 # Abtastrate (Hz)
)

# Plotte die Ergebnisse
plot_results(simulation_data)