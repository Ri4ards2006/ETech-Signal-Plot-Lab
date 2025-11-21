import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown

# Hinweis für Jupyter-User: Für interaktive Plots (Zoom/Mouseover) führe vorher %matplotlib widget aus
# matplotlib.use('Qt5Agg')  # Optional, falls der Code außerhalb von Jupyter (z. B. Skript) ausgeführt wird

def compute_signals(t, Rf, Rin, Vin_amp, Vin_freq, waveform):
    """Berechnet Eingangssignal (Vin) und ideale Ausgangssignal (Vout_ideal) für den Inverter."""
    x = 2 * np.pi * Vin_freq * t  # Phase für sinusförmige Signale
    if waveform == 'Sine':
        Vin = Vin_amp * np.sin(x)
    elif waveform == 'Square':
        Vin = Vin_amp * np.where(np.sin(x) >= 0, 1, -1)  # Quadratwelle
    elif waveform == 'Triangle':
        Vin = Vin_amp * (2 / np.pi) * np.arcsin(np.sin(x))  # Dreieckswelle
    else:
        Vin = np.zeros_like(t)  # Fallback
    
    gain = - Rf / Rin  # Theoretischer Gain (ohne Saturation)
    Vout_ideal = gain * Vin  # Ideale Ausgangsspannung
    return Vin, Vout_ideal, gain

def plot_op_amp(Rf=1e3, Rin=1e3, Vin_amp=5.0, Vin_freq=1.0, t_max=10.0, 
               waveform='Sine', Vcc=15.0):
    """Plottet Eingang- und Ausgangssignal eines Op-Amp-Inverters mit Saturation-Analyse."""
    t = np.linspace(0, t_max, 1000)  # Zeitachse (1000 Punkte, glatte Kurve)
    Vin, Vout_ideal, gain = compute_signals(t, Rf, Rin, Vin_amp, Vin_freq, waveform)
    
    # Saturation anwenden (Ausgangsklemmen auf ±Vcc)
    Vout_clamped = np.clip(Vout_ideal, -Vcc, Vcc)
    saturation = np.any(Vout_ideal > Vcc) or np.any(Vout_ideal < -Vcc)  # Sättigung-Check
    
    # Berechne P2P-Amplituden (Peak-to-Peak)
    Vin_p2p = 2 * Vin_amp  # Eingang: symmetrisch, also 2*Amp
    Vout_ideal_p2p = 2 * np.max(np.abs(Vout_ideal)) if gain != 0 else 0  # Ausgang ideal (ohne Saturation)
    Vout_clamped_p2p = 2 * np.max(np.abs(Vout_clamped)) if saturation else Vout_ideal_p2p  # Ausgang mit Saturation
    
    # Erstelle Plot-Figur mit modernem Design
    plt.figure(figsize=(12, 6), facecolor='#f5f5f5')  # Hintergrund hellgrau
    plt.subplot2grid((1, 1), (0, 0), facecolor='white')  # Plot-Bereich weiß
    
    # Eingangssignal plotten (blau, fester Strich)
    plt.plot(t, Vin, 
             label=f'Eingang ({waveform})\nAmp={Vin_amp:.1f}V | f={Vin_freq:.1f}Hz', 
             color='#2c7fb8', linewidth=2.5, alpha=0.9, zorder=1)
    
    # Ausgangssignal plotten (rot, gestrichelter Strich bei Sättigung)
    if saturation:
        # Klampiertes Signal (rot, dicker, gestrichelter Strich)
        plt.plot(t, Vout_clamped, 
                 label=f'Ausgang (gesättigt)\nP2P={Vout_clamped_p2p:.1f}V', 
                 color='#e53434', linewidth=3, linestyle='--', alpha=0.8, zorder=2)
        # Ideales Signal (dunkler rot, dünner, durchgestrichener Strich)
        plt.plot(t, Vout_ideal, 
                 color='#8e0808', linewidth=1.2, linestyle=':', alpha=0.6, 
                 label='Ausgang (ideal, unbeschränkt)')
    else:
        # Keine Sättigung → rein ausgangs-signal plotten
        plt.plot(t, Vout_clamped, 
                 label=f'Ausgang\nGain={gain:.2f} | P2P={Vout_ideal_p2p:.1f}V', 
                 color='#e53434', linewidth=3, linestyle='--', alpha=0.8, zorder=2)
    
    # Markiere Phaseninversion (Peak-Erkennung je nach Wellentyp)
    try:
        if waveform == 'Sine':
            # Lokaler Positiver Peak (Sine-Welle)
            vin_peak_mask = (Vin > 0) & (np.roll(Vin, -1) < Vin)  # Maxima mit np.roll
            if np.any(vin_peak_mask):
                idx_peak = np.argmax(vin_peak_mask)  # Ersten Peak finden
                t_peak = t[idx_peak]
                vin_peak_val = Vin[idx_peak]
                vout_peak_val = Vout_clamped[idx_peak]
        elif waveform == 'Square':
            # Mittelpunkt des ersten positiven Plateaus (Quadratwelle)
            t_peak = 0.25 / Vin_freq  # 0.25 * Periode (Periode=1/f)
            if t_peak <= t_max:
                idx_peak = np.argmin(np.abs(t - t_peak))  # Nächster Index zu t_peak
                vin_peak_val = Vin_amp
                vout_peak_val = Vout_clamped[idx_peak]
        elif waveform == 'Triangle':
            # Erster Positiver Peak (Dreieckswelle)
            t_peak = 1 / (4 * Vin_freq)  # ¼ Periode (Periode=1/f)
            if t_peak <= t_max:
                idx_peak = np.argmin(np.abs(t - t_peak))  # Nächster Index zu t_peak
                vin_peak_val = Vin_amp
                vout_peak_val = Vout_clamped[idx_peak]
        
        # Markiere Peaks, falls gefunden
        if 't_peak' in locals() and 'vin_peak_val' in locals() and 'vout_peak_val' in locals():
            # Grüner Punkt für Eingang-Peak
            plt.scatter(t_peak, vin_peak_val, s=100, color='#4caf50', zorder=3,
                        label='Vin-Peak (Phasenreferenz)')
            # Roter Punkt für Ausgangs-Peak (invers)
            plt.scatter(t_peak, vout_peak_val, s=100, color='#b71c1c', zorder=3,
                        label='Vout-Peak (invers)')
    except:
        pass  # Keine Peaks gefunden (z. B. Vin_amp=0 oder zu niedrige Frequenz)

    # Plot-Beschriftungen und Formatierung
    plt.title(f'Op-Amp Inverter-Simulation (Ideealer Op-Amp)\n'
              f'Rf={Rf:.0f}Ω | Rin={Rin:.0f}Ω | Vcc=±{Vcc:.0f}V', 
              fontsize=16, pad=20, color='#2c3e50')
    plt.xlabel('Zeit (s)', fontsize=14, color='#34495e')
    plt.ylabel('Spannung (V)', fontsize=14, color='#34495e')
    
    # Grid und Ticks (beide Major/Minor)
    plt.grid(True, linestyle='--', alpha=0.4, which='both', color='#bdc3c7')
    plt.minorticks_on()
    
    # Legende (Übersichtlich mit半透明 Hintergrund)
    plt.legend(fontsize=10, facecolor='white', edgecolor='#ecf0f1', 
               framealpha=0.9, title=f'Legende', title_fontsize=11)
    
    # Annotation mit zentralen Metriken (rechte Oberseite)
    plt.text(0.95, 0.95,  # Position (95% x, 95% y im Axes-System)
             f'Metriken:\n'
             f'- Gain (theor.): {gain:.2f}\n'
             f'- Eingang P2P: {Vin_p2p:.1f}V\n'
             f'- Ausgang P2P (gesättigt): {Vout_clamped_p2p:.1f}V\n'
             f'- Sättigung aktiv: {"✅" if saturation else "❌"}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='#e0e0e0', 
                       boxstyle='round,pad=0.5'),
             fontsize=10, va='top', ha='right', color='#666666')
    
    # Achsen-Beschränkungen (Zeit und Spannung)
    plt.xlim(0, t_max)
    plt.ylim(-Vcc*1.1, Vcc*1.1)  # Mehr Platz für Annotationen
    
    # Plot optimieren und anzeigen
    plt.tight_layout(pad=3)
    plt.show()

# Interaktive Steuerungselemente (optimierte Slider/Dropdown)
interact(plot_op_amp,
         Rf=FloatSlider(min=100, max=10e3, step=100, value=1e3, 
                       description='Rf [Ω]', continuous_update=False, 
                       readout_format='.0f', style={'description_width': 'initial'}),
         Rin=FloatSlider(min=100, max=10e3, step=100, value=1e3, 
                        description='Rin [Ω]', continuous_update=False, 
                        readout_format='.0f', style={'description_width': 'initial'}),
         Vin_amp=FloatSlider(min=0.1, max=10, step=0.1, value=5, 
                            description='Eingang-Amplitude [V]', continuous_update=False,
                            readout_format='.1f', style={'description_width': 'initial'}),
         Vin_freq=FloatSlider(min=0.1, max=10, step=0.1, value=1, 
                             description='Eingang-Frequenz [Hz]', continuous_update=False,
                             readout_format='.1f', style={'description_width': 'initial'}),
         t_max=FloatSlider(min=1, max=20, step=0.5, value=10, 
                          description='Simulationszeit [s]', continuous_update=False,
                          readout_format='.1f', style={'description_width': 'initial'}),
         waveform=Dropdown(options=['Sine', 'Square', 'Triangle'], value='Sine',
                           description='Wellentyp', style={'description_width': 'initial'}),
         Vcc=FloatSlider(min=5, max=30, step=1, value=15, 
                        description='Versorgungsspannung Vcc [V]', continuous_update=False,
                        readout_format='.0f', style={'description_width': 'initial'}));