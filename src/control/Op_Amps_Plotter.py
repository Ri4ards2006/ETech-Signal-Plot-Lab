import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def op_amp_inverter(t, Rf, Rin, Vin_amp, Vin_freq):
    """Berechnet Eingang- und Ausgangsspannung eines idealen Inverter-Schaltbilds."""
    # Eingangsspannung: Sinuswellen mit angegebener Amplitude und Frequenz
    Vin = Vin_amp * np.sin(2 * np.pi * Vin_freq * t)
    # Ausgangsspannung: Übertragungsfunktion des Inverters
    Vout = - (Rf / Rin) * Vin
    return Vin, Vout

def plot_op_amp(Rf=1e3, Rin=1e3, Vin_amp=5.0, Vin_freq=1.0, t_max=10.0):
    """Plottet Eingang- und Ausgangssignal des Inverters über Zeit."""
    t = np.linspace(0, t_max, 1000)  # Zeitachse (0 bis t_max, 1000 Punkte)
    Vin, Vout = op_amp_inverter(t, Rf, Rin, Vin_amp, Vin_freq)
    
    # Plot-Einstellungen
    plt.figure(figsize=(10, 5))
    plt.plot(t, Vin, label=f'Eingang (Vin: Amp={Vin_amp:.1f}V, f={Vin_freq:.1f}Hz)', 
             color="#2c7fb8", linewidth=2)
    plt.plot(t, Vout, label=f'Ausgang (Vout: Gain=-{Rf/Rin:.2f})', 
             color="#f15854", linewidth=1.5, linestyle='--')
    
    plt.title('Op-Amp Inverter-Simulation (idealer Op-Amp)', fontsize=14)
    plt.xlabel('Zeit (s)')
    plt.ylabel('Spannung (V)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Interaktive Slider für Parameteranpassung
interact(plot_op_amp,
         Rf=FloatSlider(min=100, max=10e3, step=100, value=1e3, description='Rf [Ω]'),
         Rin=FloatSlider(min=100, max=10e3, step=100, value=1e3, description='Rin [Ω]'),
         Vin_amp=FloatSlider(min=0.1, max=10, step=0.1, value=5, description='Vin-Amplitude [V]'),
         Vin_freq=FloatSlider(min=0.1, max=10, step=0.1, value=1, description='Vin-Frequenz [Hz]'),
         t_max=FloatSlider(min=1, max=20, step=0.5, value=10, description='Simulationszeit [s]'));