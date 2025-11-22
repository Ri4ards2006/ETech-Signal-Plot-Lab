import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown

def compute_signals(t, Vin_amp, Vin_freq, waveform):
    """Berechnet Eingangssignal (Vin)."""
    x = 2 * np.pi * Vin_freq * t
    if waveform == 'Sine':
        Vin = Vin_amp * np.sin(x)
    elif waveform == 'Square':
        Vin = Vin_amp * np.where(np.sin(x) >= 0, 1, -1)
    elif waveform == 'Triangle':
        Vin = Vin_amp * (2 / np.pi) * np.arcsin(np.sin(x))
    else:
        Vin = np.zeros_like(t)
    return Vin

def plot_comparator(Vin_amp=5.0, Vin_freq=1.0, t_max=10.0,
                    waveform='Sine', Vcc=15.0, Vref=0.0):
    """Simuliert einen idealen Komparator mit einstellbarer Referenzspannung."""
    
    t = np.linspace(0, t_max, 2000)
    Vin = compute_signals(t, Vin_amp, Vin_freq, waveform)

    # ---------- Komparator Kern ----------
    # Wenn Vin > Vref → +Vcc
    # Wenn Vin < Vref → -Vcc
    Vout = np.where(Vin > Vref, Vcc, -Vcc)
    # -------------------------------------

    plt.figure(figsize=(12, 6))

    # Eingangsbereich
    plt.plot(t, Vin, label='Eingangssignal Vin', color='#2c7fb8', linewidth=2)

    # Referenzlinie
    plt.axhline(Vref, color='orange', linestyle='--',
                label=f'Referenzspannung Vref = {Vref:.2f} V')

    # Ausgangssignal
    plt.plot(t, Vout, label='Komparator-Ausgang', color='#e53434', linewidth=2)

    plt.title('Ideal-Komparator Simulation')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Spannung (V)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(-Vcc * 1.2, Vcc * 1.2)
    plt.tight_layout()
    plt.show()

# Interaktive UI
interact(plot_comparator,
         Vin_amp=FloatSlider(min=0.1, max=10, step=0.1, value=5,
                             description='Eingangs-Amplitude [V]'),
         Vin_freq=FloatSlider(min=0.1, max=10, step=0.1, value=1,
                              description='Eingangs-Frequenz [Hz]'),
         t_max=FloatSlider(min=1, max=20, step=0.5, value=10,
                           description='Simulationszeit [s]'),
         waveform=Dropdown(options=['Sine', 'Square', 'Triangle'], value='Sine',
                           description='Wellenform'),
         Vcc=FloatSlider(min=5, max=30, step=1, value=15,
                         description='Vcc Versorgung [V]'),
         Vref=FloatSlider(min=-10, max=10, step=0.1, value=0,
                          description='Referenzspannung Vref [V]'));
