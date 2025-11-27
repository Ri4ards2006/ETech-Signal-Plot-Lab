import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- GUI-Einrichtung ---
plt.ion()  # Interactive On

fig, (ax_bode, ax_step) = plt.subplots(2, 1, figsize=(10, 8))
canvas_bode = fig.canvas
canvas_step = fig.canvas

def update_plots():
    # Eingabefenster öffnen, um R, C und Filter-Typ zu erhalten (manuelle Eingabe)
    # Beachte: Standardwerte werden bei leerer Eingabe verwendet
    R = input("Widerstand R [Ω] (Standard: 1000): ") or 1000
    C = input("Kapazität C [F] (Standard: 1e-6): ") or 1e-6
    filter_type = input("Filter-Typ (lowpass/highpass) (Standard: lowpass): ") or 'lowpass'

    # --- Parameterberechnung ---
    R = float(R)
    C = float(C)
    tau = R * C
    fc = 1 / (2 * np.pi * tau) if tau != 0 else 0

    # --- Frequenzbereich ---
    w = np.logspace(-1, 5, 500)  # Basis für Frequenzbereich (logarithmisch)
    w_rad, H = signal.freqs(num, den, w)  # Frequenzgang H(jω)
    f_hz = w_rad / (2 * np.pi)  # Umrechnung von rad/s zu Hz
    gain_db = 20 * np.log10(np.abs(H))  # Gain in dB
    phase_deg = np.angle(H, deg=True)  # Phase in Grad

    # Übertragungsfunktion basierend auf Filter-Typ berechnen
    if filter_type == 'lowpass':
        num = [1]
        den = [tau, 1]
    elif filter_type == 'highpass':
        num = [tau, 0]
        den = [tau, 1]
    else:
        num, den = [1], [1]  # Fallback für ungültige Eingaben
        print("Ungültiger Filter-Typ. Standard (lowpass) verwendet.")

    # --- Zeitbereich ---
    t = np.linspace(0, 5 * tau, 500) if tau != 0 else np.linspace(0, 1, 500)
    t_step, y_step = signal.step((num, den), T=t)  # Simulierte Sprungantwort

    # Theoretische Sprungantwort basierend auf Filter-Typ
    if filter_type == 'lowpass':
        y_theo = 1 - np.exp(-t_step / tau)
    elif filter_type == 'highpass':
        y_theo = np.exp(-t_step / tau)
    else:
        y_theo = np.ones_like(t_step)

    # --- Plots aktualisieren ---
    # Bode-Diagramm
    ax_bode.clear()
    ax_bode.semilogx(f_hz, gain_db)
    ax_bode.set_title('Bode-Diagramm')
    ax_bode.set_xlabel('Frequenz [Hz]')
    ax_bode.set_ylabel('Verstärkung [dB]')
    ax_bode.grid(True)
    ax_bode.axvline(fc, color='r', linestyle='--', label=f'Eckfrequenz $f_c$ = {fc:.2f} Hz')
    ax_bode.legend()

    # Sprungantwort
    ax_step.clear()
    ax_step.plot(t_step, y_step, label='Simulierte $V_{out}(t)$')
    ax_step.plot(t_step, y_theo, '--', label='Theoretische $V_{out}(t)$')
    ax_step.plot(t_step, np.ones_like(t_step), 'k--', alpha=0.5, label='Eingangsspannung $V_{in}(t)$ (Sprung)')
    ax_step.set_title(f'Sprungantwort ${filter_type}$-Filter')
    ax_step.set_xlabel('Zeit [s]')
    ax_step.set_ylabel('Spannung [V]')
    ax_step.grid(True)
    ax_step.legend(loc='upper right')

    # --- Plot anzeigen ---
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)  # Pause kurz einlegen, um Updates zu verbinden

# Initialer Plot bei Programmstart
update_plots()

# --- Hauptschleife ---
while True:
    # User-Wartezeit für Eingabe (Aktualisieren)
    input("Drücke Enter, um die Plots zu aktualisieren...\n")
    update_plots()

plt.ioff()  # Interactive Off, falls benötigt
plt.show()  # Ausführen, falls nötig, aber nicht im Loop nötig