# =============================================================================
# Sine & Cosine Wave Generator with Adjustable Amplitude, Frequency, and Phase
# Purpose: Generate and visualize sinusoidal signals with customizable parameters
# Author: [Your Name] | Date: [Insert Date]
# =============================================================================

# ------------------------------
# 1. Import Libraries
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider  # Für interaktive Parameter-Einstellungen (optional, aber cool!)

# ------------------------------
# 2. Define Core Parameters (Standardwerte)
# ------------------------------
# Zeitbereich
fs = 1000    # Samplingrate (Hz) → Genug, um Frequenzen bis 500 Hz zu erfassen
T = 5.0      # Gesamtbeobachtungszeit (s) → Länger = mehr Perioden sichtbar
t = np.linspace(0, T, int(fs*T), endpoint=False)  # Zeitarray (0 bis T, ohne Überschneidung)

# Standard-Signal-Parameter (wird später via Slider verändert)
A_sin = 1.5   # Amplitude Sine-Welle (V)
f_sin = 50    # Frequenz Sine-Welle (Hz)
phi_sin = 0   # Phase Sine-Welle (Radiant) → 0 = keinerlei Verschiebung

A_cos = 1.0   # Amplitude Cosine-Welle (V)
f_cos = 50    # Frequenz Cosine-Welle (Hz) → Gleich Sine-Frequenz zum Vergleich
phi_cos = 0   # Phase Cosine-Welle (Radiant) → 0 = Cosine = Sine mit 90° Verschiebung

# ------------------------------
# 3. Signal-Generierungsfunktion
# ------------------------------
def generate_sinusoid(t, A, f, phi, waveform_type='sin'):
    """
    Generiert ein Sinus- oder Kosinus-Signal mit gegebenen Parametern.
    
    Args:
        t (np.array): Zeitachse (s)
        A (float): Amplitude (V)
        f (float): Frequenz (Hz)
        phi (float): Phase (Radiant)
        waveform_type (str): 'sin' oder 'cos' (Standard: 'sin')
    
    Returns:
        np.array: Generiertes Signal
    """
    omega = 2 * np.pi * f  # Kreisfrequenz (rad/s)
    if waveform_type == 'sin':
        return A * np.sin(omega * t + phi)
    elif waveform_type == 'cos':
        return A * np.cos(omega * t + phi)
    else:
        raise ValueError("waveform_type must be 'sin' or 'cos'")

# ------------------------------
# 4. Initial Signal Generation
# ------------------------------
# Generiere Sine und Cosine mit Standardparametern
signal_sin = generate_sinusoid(t, A_sin, f_sin, phi_sin, 'sin')
signal_cos = generate_sinusoid(t, A_cos, f_cos, phi_cos, 'cos')

# ------------------------------
# 5. Plot Setup (Interaktiv mit Sliders!)
# ------------------------------
# Erzeuge eine Figur mit Subplots und Platz für Sliders
fig, ax = plt.subplots(figsize=(12, 8))  # Großes Fenster für Clarity
plt.subplots_adjust(left=0.1, bottom=0.35)  # Platz unten für Sliders

# Plotte die Signale im Zeitbereich
line_sin, = plt.plot(t, signal_sin, color='blue', linewidth=2, label=f'Sine (A={A_sin}, f={f_sin} Hz)')
line_cos, = plt.plot(t, signal_cos, color='red', linewidth=1.5, linestyle='--', 
                    label=f'Cosine (A={A_cos}, f={f_cos} Hz, φ={phi_cos:.1f} rad)')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (V)', fontsize=12)
plt.title('Adjustable Sine & Cosine Waves', fontsize=14, fontweight='bold')
plt.xlim(0, T)  # Zeitausschnitt auf 0–5s
plt.ylim(-max(A_sin, A_cos)*1.2, max(A_sin, A_cos)*1.2)  # Y-Achse mit Puffer
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# ------------------------------
# 6. Erstelle Sliders für Parameter-Einstellungen (Interaktivität!)
# ------------------------------
# Slider für Amplitude Sine
ax_A_sin = plt.axes([0.1, 0.25, 0.7, 0.03])  # [x, y, width, height]
slider_A_sin = Slider(
    ax=ax_A_sin,
    label='Sine Amplitude (A):',
    valmin=0.1,  # Min. Amplitude
    valmax=3.0,  # Max. Amplitude
    valinit=A_sin,  # Startwert
    color='blue'
)

# Slider für Frequenz Sine
ax_f_sin = plt.axes([0.1, 0.20, 0.7, 0.03])
slider_f_sin = Slider(
    ax=ax_f_sin,
    label='Sine Frequency (f):',
    valmin=10,  # Min. Frequenz (Hz)
    valmax=200,  # Max. Frequenz (Hz)
    valinit=f_sin,
    color='blue'
)

# Slider für Phase Sine
ax_phi_sin = plt.axes([0.1, 0.15, 0.7, 0.03])
slider_phi_sin = Slider(
    ax=ax_phi_sin,
    label='Sine Phase (φ):',
    valmin=-np.pi,  # Min. Phase (-π rad)
    valmax=np.pi,   # Max. Phase (π rad)
    valinit=phi_sin,
    color='blue'
)

# Slider für Amplitude Cosine
ax_A_cos = plt.axes([0.1, 0.10, 0.7, 0.03])
slider_A_cos = Slider(
    ax=ax_A_cos,
    label='Cosine Amplitude (A):',
    valmin=0.1,
    valmax=3.0,
    valinit=A_cos,
    color='red'
)

# Slider für Phase Cosine (Frequenz_cos = Frequenz_sin für einfacheres Vergleich!)
ax_phi_cos = plt.axes([0.1, 0.05, 0.7, 0.03])
slider_phi_cos = Slider(
    ax=ax_phi_cos,
    label='Cosine Phase (φ):',
    valmin=-np.pi,
    valmax=np.pi,
    valinit=phi_cos,
    color='red'
)

# ------------------------------
# 7. Update-Funktion für Sliders (Reaktion auf Einstellungen)
# ------------------------------
def update(val):
    # Sine-Parameter aktualisieren
    A_sin_new = slider_A_sin.val
    f_sin_new = slider_f_sin.val
    phi_sin_new = slider_phi_sin.val
    # Generiere neues Sine-Signal
    signal_sin_new = generate_sinusoid(t, A_sin_new, f_sin_new, phi_sin_new, 'sin')
    # Plot aktualisieren
    line_sin.set_ydata(signal_sin_new)
    line_sin.set_label(f'Sine (A={A_sin_new:.1f}, f={f_sin_new:.0f} Hz)')
    
    # Cosine-Parameter aktualisieren (Frequenz.Cos = Frequenz.Sin)
    A_cos_new = slider_A_cos.val
    phi_cos_new = slider_phi_cos.val
    # Generiere neues Cosine-Signal (Frequenz fix auf Sine-Frequenz!)
    signal_cos_new = generate_sinusoid(t, A_cos_new, f_sin_new, phi_cos_new, 'cos')
    # Plot aktualisieren
    line_cos.set_ydata(signal_cos_new)
    line_cos.set_label(f'Cosine (A={A_cos_new:.1f}, f={f_sin_new:.0f} Hz, φ={phi_cos_new:.1f} rad)')
    
    # Legende neu rendern (damit Labels aktualisiert werden)
    plt.legend()
    # Zeichne Plot neu
    fig.canvas.draw_idle()

# ------------------------------
# 8. Sliders an Update-Funktion binden
# ------------------------------
slider_A_sin.on_changed(update)
slider_f_sin.on_changed(update)
slider_phi_sin.on_changed(update)
slider_A_cos.on_changed(update)
slider_phi_cos.on_changed(update)

# ------------------------------
# 9. Anzeige des Plots
# ------------------------------
plt.show()