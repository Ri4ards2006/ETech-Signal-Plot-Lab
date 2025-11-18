# =============================================================================
# Square & Triangle Wave Generator with Adjustable Parameters
# Purpose: Generate and visualize square/triangle waves with customizable amplitude, frequency, phase, and duty cycle
# Author: [Your Name] | Date: [Insert Date]
# =============================================================================

# ------------------------------
# 1. Import Libraries
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider  # Für interaktive Parameter-Einstellungen
from scipy.signal import square, sawtooth  # Scipy-Funktionen für Wellen-Generierung

# ------------------------------
# 2. Define Core Parameters (Standardwerte)
# ------------------------------
# Zeitbereich
fs = 1000    # Samplingrate (Hz) → Genug, um Frequenzen bis 500 Hz zu erfassen
T = 4.0      # Gesamtbeobachtungszeit (s) → Länger = mehr Perioden sichtbar
t = np.linspace(0, T, int(fs*T), endpoint=False)  # Zeitarray (0 bis T, ohne Überschneidung)

# Standard-Parameter für Rechteckwelle
A_sq = 2.0    # Amplitude (Peak-to-Peak, V) → Rechteckwellen haben typischerweise P-P-Amplitude
f_sq = 25     # Frequenz (Hz)
phi_sq = 0    # Phase (Radiant) → Verschiebung der ersten Flanke
duty_sq = 0.5 # Duty-Cycle (Fraktion) → 0.5 = 50% High / 50% Low (Symmetrisch)

# Standard-Parameter für Dreieckwelle
A_tri = 1.5   # Amplitude (Peak-to-Peak, V)
f_tri = 25    # Frequenz (Hz) → Gleiche Frequenz wie Rechteckwelle für Vergleich
phi_tri = 0   # Phase (Radiant)

# ------------------------------
# 3. Signal-Generierungsfunktionen
# ------------------------------
def generate_square(t, A, f, phi, duty):
    """
    Generiert eine Rechteckwelle mit gegebenen Parametern.
    
    Args:
        t (np.array): Zeitachse (s)
        A (float): Amplitude (Peak-to-Peak, V)
        f (float): Frequenz (Hz)
        phi (float): Phase (Radiant) → Verschiebt die Wellenform entlang der Zeitachse
        duty (float): Duty-Cycle (0 < duty < 1) → Fraktion der Periode, in der das Signal High ist
    
    Returns:
        np.array: Generierte Rechteckwelle
    """
    # Kreisfrequenz (ω = 2πf) für Phase-Berechnung
    omega = 2 * np.pi * f
    # Phase-Adjustierung: Füge die Phase zur Zeit ein (φ wird in rad verwendet)
    phase_shifted_t = omega * t + phi
    # Scipy's square-Funktion: Generiert Rechteckwelle mit Duty-Cycle
    #   → square(x, duty) → x = ωt + φ, duty = High-Time / Period
    #   → Output: [-1, 1] bei Standard → Skalieren zu [ -A/2, A/2 ] (Peak-to-Peak)
    signal_sq = A * 0.5 * square(phase_shifted_t, duty)  # 0.5 skaliert von [-1,1] zu [-0.5A, 0.5A]
    return signal_sq

def generate_triangle(t, A, f, phi):
    """
    Generiert eine Dreieckwelle mit gegebenen Parametern.
    
    Args:
        t (np.array): Zeitachse (s)
        A (float): Amplitude (Peak-to-Peak, V)
        f (float): Frequenz (Hz)
        phi (float): Phase (Radiant) → Verschiebt die Wellenform entlang der Zeitachse
    
    Returns:
        np.array: Generierte Dreieckwelle
    """
    # Kreisfrequenz und Phasenverschiebung (如同 Sine-Welle)
    omega = 2 * np.pi * f
    phase_shifted_t = omega * t + phi
    # Scipy's sawtooth-Funktion mit width=0.5 → Dreieckwelle (Standard sawtooth ist Sägezahn)
    #   → sawtooth(x, width) → x = ωt + φ, width=0.5 → Aufstieg und Abstieg je 50% der Periode
    #   → Output: [-1, 1] bei Standard → Skalieren zu [ -A/2, A/2 ]
    signal_tri = A * 0.5 * sawtooth(phase_shifted_t, width=0.5)
    return signal_tri

# ------------------------------
# 4. Generiere Startsignale mit Standardparametern
# ------------------------------
signal_sq = generate_square(t, A_sq, f_sq, phi_sq, duty_sq)
signal_tri = generate_triangle(t, A_tri, f_tri, phi_tri)

# ------------------------------
# 5. Plot Setup (Interaktiv mit Sliders!)
# ------------------------------
# Erzeuge Figur und Axes mit Platz für Sliders unten
fig, ax = plt.subplots(figsize=(12, 8))  # Großes Fenster für Clarity
plt.subplots_adjust(left=0.1, bottom=0.4)  # Platz für 4 Sliders (ca. 0.4 von unten)
ax.set_position([0.1, 0.55, 0.8, 0.4])  # Plot-Bereich (x, y, width, height)

# Plotte beide Signale im Zeitbereich
line_sq, = ax.plot(t, signal_sq, color='green', linewidth=2, 
                  label=f'Square (A={A_sq:.1f}, f={f_sq:.0f} Hz, φ={phi_sq:.1f} rad, Duty={duty_sq:.1f})')
line_tri, = ax.plot(t, signal_tri, color='purple', linewidth=1.5, linestyle='--', 
                   label=f'Triangle (A={A_tri:.1f}, f={f_tri:.0f} Hz, φ={phi_tri:.1f} rad)')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Amplitude (V)', fontsize=12)
ax.set_title('Adjustable Square & Triangle Waves', fontsize=14, fontweight='bold')
ax.set_xlim(0, T)  # Zeitausschnitt auf 0–4s
ax.set_ylim(-max(A_sq, A_tri)*1.1, max(A_sq, A_tri)*1.1)  # Y-Achse mit Puffer
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# ------------------------------
# 6. Erstelle Sliders für Parameter-Einstellungen
# ------------------------------

# Slider für Rechteckwelle-Amplitude
ax_A_sq = plt.axes([0.1, 0.40, 0.7, 0.03])  # [x, y, width, height]
slider_A_sq = Slider(
    ax=ax_A_sq,
    label='Square Amplitude (A):',
    valmin=0.2,  # Min. Amplitude (V)
    valmax=3.0,  # Max. Amplitude (V)
    valinit=A_sq,
    color='green'
)

# Slider für Frequenz (gemeinsam für beide Wellen, einfacher Vergleich)
ax_f = plt.axes([0.1, 0.35, 0.7, 0.03])
slider_f = Slider(
    ax=ax_f,
    label='Frequency (f):',
    valmin=10,   # Min. Frequenz (Hz)
    valmax=150,  # Max. Frequenz (Hz)
    valinit=f_sq,
    color='orange'  # Kontrastfarbe für Frequenz-Slider
)

# Slider für Phase (gemeinsam für beide Wellen)
ax_phi = plt.axes([0.1, 0.30, 0.7, 0.03])
slider_phi = Slider(
    ax=ax_phi,
    label='Phase (φ):',
    valmin=-np.pi,  # Min. Phase (-π rad)
    valmax=np.pi,   # Max. Phase (π rad)
    valinit=phi_sq,
    color='blue'    # Kontrastfarbe für Phase-Slider
)

# Slider für Rechteckwelle-Duty-Cycle
ax_duty = plt.axes([0.1, 0.25, 0.7, 0.03])
slider_duty = Slider(
    ax=ax_duty,
    label='Square Duty-Cycle:',
    valmin=0.1,  # Min. Duty (10% High)
    valmax=0.9,  # Max. Duty (90% High)
    valinit=duty_sq,
    color='green'  # Matchet Rechteckwelle-Farbe
)

# Slider für Dreieckwelle-Amplitude (optional, falls Du sie variabel haben möchtest)
# (Auskommentiert, weil Standard-Pfad "A_sq und A_tri getrennt" ist – aktiviere, wenn gewünscht)
# ax_A_tri = plt.axes([0.1, 0.20, 0.7, 0.03])
# slider_A_tri = Slider(...)

# ------------------------------
# 7. Update-Funktion für Sliders (Reaktion auf Einstellungen)
# ------------------------------
def update(val):
    # Aktualisiere Parameter
    A_sq_new = slider_A_sq.val
    f_new = slider_f.val
    phi_new = slider_phi.val
    duty_new = slider_duty.val
    
    # Generiere neue Signale mit aktualisierten Parametern
    # Rechteckwelle
    signal_sq_new = generate_square(t, A_sq_new, f_new, phi_new, duty_new)
    line_sq.set_ydata(signal_sq_new)
    # Aktualisiere Label mit neuen Werten
    line_sq.set_label(f'Square (A={A_sq_new:.1f}, f={f_new:.0f} Hz, φ={phi_new:.1f} rad, Duty={duty_new:.1f})')
    
    # Dreieckwelle (Frequenz und Phase von Rechteckwelle übernommen, für direkten Vergleich)
    signal_tri_new = generate_triangle(t, A_tri, f_new, phi_new)  # A_tri bleibt Standard (oder passe zu slider_A_tri)
    line_tri.set_ydata(signal_tri_new)
    line_tri.set_label(f'Triangle (A={A_tri:.1f}, f={f_new:.0f} Hz, φ={phi_new:.1f} rad)')
    
    # Y-Achse skaliert neu (falls Amplitude von Rechteckwelle verändert wurde)
    current_max_amp = max(A_sq_new, A_tri)
    ax.set_ylim(-current_max_amp*1.1, current_max_amp*1.1)
    
    # Legende und Plot aktualisieren
    ax.legend()
    fig.canvas.draw_idle()

# ------------------------------
# 8. Sliders an Update-Funktion binden
# ------------------------------
slider_A_sq.on_changed(update)
slider_f.on_changed(update)
slider_phi.on_changed(update)
slider_duty.on_changed(update)

# ------------------------------
# 9. Anzeige des Plots
# ------------------------------
plt.show()