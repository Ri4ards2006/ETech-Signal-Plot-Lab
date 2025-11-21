import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown
matplotlib.use('Qt5Agg')  # Backend für interaktive GUI-Plots (wenn nicht in Jupyter)

def compute_signals(t, Rf, Rin, Vin_amp, Vin_freq, waveform):
    """Berechnet Eingangssignal (Vin) und ideale Ausgangssignal (Vout_ideal) für den Inverter."""
    x = 2 * np.pi * Vin_freq * t  # Phase für sinusförmige Signale
    if waveform == 'Sine':
        Vin = Vin_amp * np.sin(x)
    elif waveform == 'Square':
        Vin = Vin_amp * np.where(np.sin(x) >= 0, 1, -1)  # Quadratwellen mit Schwelle
    elif waveform == 'Triangle':
        Vin = Vin_amp * (2 / np.pi) * np.arcsin(np.sin(x))  # Dreieckswelle über arcsin
    else:
        Vin = np.zeros_like(t)  # Fallback (nicht nötig, wenn Wellenformern korrekt definiert)
    
    # Ideale Ausgangsspannung (ohne Saturation)
    gain = - Rf / Rin
    Vout_ideal = gain * Vin
    return Vin, Vout_ideal, gain

def plot_op_amp(Rf=1e3, Rin=1e3, Vin_amp=5.0, Vin_freq=1.0, t_max=10.0, 
               waveform='Sine', Vcc=15.0):
    """Plottet Eingang- und Ausgangssignal eines Op-Amp-Inverters mit Saturation-Analyse."""
    t = np.linspace(0, t_max, 1000)  # Zeitachse
    Vin, Vout_ideal, gain = compute_signals(t, Rf, Rin, Vin_amp, Vin_freq, waveform)
    
    # Saturation anwenden (realistische Op-Amp-Grenze)
    Vout_clamped = np.clip(Vout_ideal, -Vcc, Vcc)
    saturation = np.any(Vout_ideal > Vcc) or np.any(Vout_ideal < -Vcc)  # Check for clipping
    
    # Signalparameter berechnen
    Vin_p2p = 2 * Vin_amp  # Eingang Peak-to-Peak (da symmetrisch)
    Vout_ideal_p2p = 2 * np.max(np.abs(Vout_ideal))
    Vout_clamped_p2p = 2 * np.max(np.abs(Vout_clamped)) if saturation else Vout_ideal_p2p
    
    # Plot-Einstellungen mit moderner Optik
    plt.figure(figsize=(12, 6), facecolor='lightgray')
    plt.subplot2grid((1, 1), (0, 0), facecolor='white')  # Hintergrund für Plot-Area
    
    # Eingangssignal plotten
    plt.plot(t, Vin, 
             label=f'Eingang ({waveform})\nAmp={Vin_amp:.1f}V | f={Vin_freq:.1f}Hz | P2P={Vin_p2p:.1f}V', 
             color='#2c7fb8', linewidth=2, alpha=0.9, zorder=1)
    
    # Ausgangssignal plotten (mit Saturation-Markierung)
    if saturation:
        # Klampiertes Signal (starker, fester Stil)
        plt.plot(t, Vout_clamped, 
                 label=f'Ausgang (gesättigt)\nGain={gain:.2f} | Vout P2P={Vout_clamped_p2p:.1f}V', 
                 color='#f15854', linewidth=2.5, linestyle='--', alpha=0.9, zorder=2)
        # Ideales Signal (dünner, durchgestrichener Stil, um Clipping zu zeigen)
        plt.plot(t, Vout_ideal, 
                 color='#8e0808', linewidth=1.2, linestyle=':', alpha=0.7, 
                 label='Vout Ideal (außerhalb Vcc)')
    else:
        # Keine Saturation → rein ideales Signal plotten
        plt.plot(t, Vout_clamped, 
                 label=f'Ausgang\nGain={gain:.2f} | Vout P2P={Vout_ideal_p2p:.1f}V', 
                 color='#f15854', linewidth=2.5, linestyle='--', alpha=0.9, zorder=2)
    
    # Phaseninversion hervorheben (markiere Peaks)
    # Finde erste positive Vin-Peak-Position
    try:
        vin_peak_mask = (Vin > 0) & (np.roll(Vin, -1) < Vin)  # Lokale Maxima
        if np.any(vin_peak_mask):
            t_peak = t[vin_peak_mask][0]
            vin_peak_val = Vin[vin_peak_mask][0]
            vout_peak_val = Vout_clamped[vin_peak_mask][0]
            
            plt.scatter(t_peak, vin_peak_val, s=80, color='lime', zorder=3, 
                        label='Vin-Peak (Phasenreferenz)')
            plt.scatter(t_peak, vout_peak_val, s=80, color='darkred', zorder=3, 
                        label='Vout-Peak (invers)')
    except IndexError:
        pass  # Kein Peak gefunden (z. B. zu niedrige Frequenz)
    
    # Titel und Achsenbeschriftungen
    plt.title(f'Op-Amp Inverter-Simulation (Ideenop-Amp mit Saturation)\nRf={Rf:.0f}Ω | Rin={Rin:.0f}Ω | Vcc=±{Vcc:.0f}V', 
              fontsize=16, pad=20)
    plt.xlabel('Zeit (s)', fontsize=14)
    plt.ylabel('Spannung (V)', fontsize=14)
    
    # Grid und Legende
    plt.grid(True, linestyle='--', alpha=0.5, which='both')  # Major und Minor Grid
    plt.minorticks_on()
    plt.legend(fontsize=10, facecolor='white', edgecolor='gray', framealpha=0.8)
    
    # Annotation mit zentralen Metriken
    plt.text(0.05, 0.95, 
             f'Parameter:\n- Theoretischer Gain: {gain:.2f}\n- Eingang P2P: {Vin_p2p:.1f}V\n- Ausgang (ideal) P2P: {Vout_ideal_p2p:.1f}V\n- Saturation aktiv: {"✅" if saturation else "❌"}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'),
             fontsize=10, va='top')
    
    # Plot anpassen und anzeigen
    plt.xlim(0, t_max)
    plt.ylim(-Vcc*1.1, Vcc*1.1)  # Mehr Platz für Saturation-Annotationen
    plt.tight_layout()
    plt.show()

# Interaktive Steuerungselemente (mit erweiterten Optionen)
interact(plot_op_amp,
         Rf=FloatSlider(min=100, max=10e3, step=100, value=1e3, 
                       description='Rf [Ω]', continuous_update=False),
         Rin=FloatSlider(min=100, max=10e3, step=100, value=1e3, 
                        description='Rin [Ω]', continuous_update=False),
         Vin_amp=FloatSlider(min=0.1, max=10, step=0.1, value=5, 
                            description='Vin-Amplitude [V]', readout_format='.1f',
                            continuous_update=False),
         Vin_freq=FloatSlider(min=0.1, max=10, step=0.1, value=1, 
                             description='Vin-Frequenz [Hz]', readout_format='.1f',
                             continuous_update=False),
         t_max=FloatSlider(min=1, max=20, step=0.5, value=10, 
                          description='Simulationszeit [s]', continuous_update=False),
         waveform=Dropdown(options=['Sine', 'Square', 'Triangle'], value='Sine',
                           description='Wellentyp'),
         Vcc=FloatSlider(min=5, max=30, step=1, value=15, 
                        description='Vcc [V]', continuous_update=False));