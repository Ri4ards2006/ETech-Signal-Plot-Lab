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
        dict: Enthält Zeitachse, Originalsignal, und Doppler-effektes Signal.
    """
    # Zeitachse generieren (Zeitpunkte in Sekunden)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Phaseinkrement berücksichtigt die Bewegung des Beobachters und der Quelle
    phase_increment = (2 * np.pi * f_source) * ( (v_observer + v_source) / c )  # Formel für Doppler-Phase
    # Originalsignal (Sinus mit constant Phase-Einkrement)
    original_signal = np.sin( (2 * np.pi * f_source * t) + phase_increment * t )
    
    # Doppler-effektes Signal (moduliert mit der Bewegung)
    f_observed = f_source * (c + v_observer) / (c + v_source)
    doppler_signal = np.sin( 2 * np.pi * f_observed * t )
    
    return {
        't': t,
        'original_signal': original_signal,
        'doppler_signal': doppler_signal
    }

def plot_results(simulation_data):
    """
    Plotten der originalen und Doppler-effektes Signale im Zeitbereich und deren FFT.
    """
    # Zeitbereich-Plot
    plt.figure(figsize=(12, 6))
    plt.plot(simulation_data['t'], simulation_data['original_signal'], label='Original')
    plt.plot(simulation_data['t'], simulation_data['doppler_signal'], label='Doppler')
    plt.title('Zeitbereich - Original vs. Beobachtetes Signal')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # FFT für Frequenzspektrum
    original_fft = fft(simulation_data['original_signal'])
    doppler_fft = fft(simulation_data['doppler_signal'])
    freqs_original = fftfreq(len(simulation_data['original_signal']), 1 / simulation_data['sample_rate'])
    freqs_doppler = fftfreq(len(simulation_data['doppler_signal']), 1 / simulation_data['sample_rate'])
    
    # Frequenzbereich-Plot
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_original, np.abs(original_fft), label='Original')
    plt.plot(freqs_doppler, np.abs(doppler_fft), label='Doppler')
    plt.xlim(0, simulation_data['f_source'] * 2)  # Nur positive Frequenzen bis 2*f_source zeigen
    plt.title('Frequenzbereich - FFT des Original- und Doppler-Signals')
    plt.xlabel('Frequenz (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# Simulierung mit Beispielwerten
simulation_data = simulate_doppler_effect(
    f_source=1000,    # Originalfrequenz (kHz)
    v_observer=5,     # Beobachtergeschwindigkeit (m/s, positiv naht der Quelle)
    v_source=0,       # Quellengeschwindigkeit (m/s, 0 = ruhend)
    c=343,            # Schallgeschwindigkeit in Luft (m/s)
    duration=2,       # Simulationdauer (s)
    sample_rate=44100 # Audiosample-Rate (Hz)
)

# Plotte die Ergebnisse
plot_results(simulation_data)