import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =================================================================
# 1. SCHALTUNGSPARAMETER & BERECHNUNGEN
# =================================================================
# Schaltung: RC-Tiefpassfilter (Ausgang über C)
R = 1000    # Widerstand in Ohm
C = 1e-6    # Kapazität in Farad (1 µF)
tau = R * C  # Zeitkonstante [s]
fc = 1 / (2 * np.pi * tau)  # Eckfrequenz/Grenzfrequenz in Hz

print(f"Schaltung: RC-Tiefpassfilter")
print(f"Zeitkonstante (τ): {tau:.6f} s")
print(f"Eckfrequenz (f_c): {fc:.2f} Hz\n")

# Koeffizienten der Übertragungsfunktion H(s) = 1 / (τ*s + 1)
# Zählerpolynom (Numerator): [1] (Ausdruck: 1)
# Nennerpolynom (Denominator): [1, τ] (Ausdruck: τ*s + 1)
num = [1]
den = [1, tau]  # Korrektur der Nennerkoeffizienten

# =================================================================
# 2. BODE-DIAGRAMM (FREQUENZBEREICH)
# =================================================================
# Winkelfrequenzvektor (logarithmisch, von 0.1 bis 1e5 rad/s)
w = np.logspace(-1, 5, 500)  
# Frequenzgang H(jω) berechnen (w_rad = w, H = komplexe Übertragung)
w_rad, H = signal.freqs(num, den, w)  

# Umrechnung auf Frequenz f [Hz]
f_hz = w_rad / (2 * np.pi)

# Gain in dB und Phase in Grad berechnen
gain_db = 20 * np.log10(np.abs(H))
phase_deg = np.angle(H, deg=True)  # Phase zwischen -180° und 180°

# Bode-Diagramm zeichnen
plt.figure(figsize=(10, 8))

# --- Subplot 1: Gain ---
plt.subplot(2, 1, 1)
plt.semilogx(f_hz, gain_db)
plt.title('Bode-Diagramm des RC-Tiefpassfilters')
plt.xlabel('Frequenz [Hz]')  # Hinzugefügte x-Achsenbezeichnung
plt.ylabel('Verstärkung [dB]')
plt.grid(which='both', axis='both', linestyle='--', alpha=0.7)
plt.axvline(fc, color='r', linestyle='--', label=f'Eckfrequenz $f_c$ = {fc:.2f} Hz')
plt.legend()
# y-Achsenlimit anpassen, um den relevanten Bereich besser sichtbar zu machen
plt.ylim(top=5, bottom=-60)

# --- Subplot 2: Phase ---
plt.subplot(2, 1, 2)
plt.semilogx(f_hz, phase_deg)
plt.xlabel('Frequenz [Hz]')
plt.ylabel('Phase [°]')
plt.grid(which='both', axis='both', linestyle='--', alpha=0.7)
plt.axvline(fc, color='r', linestyle='--', label=f'Eckfrequenz $f_c$ = {fc:.2f} Hz')
plt.legend()
# Phase-Limit festlegen (von -90° bis 0°, typisch für Tiefpass)
plt.ylim(-90, 0)

plt.tight_layout()  # Besserer Layout für Subplots
plt.show()

# =================================================================
# 3. ZEITBEREICHS-SIMULATION (SPRUNGANTWORT)
# =================================================================
# Zeitvektor (0 bis 5*τ, 500 Punkte)
t = np.linspace(0, 5 * tau, 500)

# Sprungantwort berechnen (System als Tupel (num, den) übergeben)
t_step, y_step = signal.step((num, den), T=t)  # Korrekte Übergabe des Systems

# Sprungantwort plotte
plt.figure(figsize=(8, 4))
plt.plot(t_step, y_step, label='$V_{out}(t)$ (Gefiltert)')
plt.plot(t_step, np.ones_like(t_step), 'k--', alpha=0.5, label='$V_{in}(t)$ (Eingangssprung)')
plt.title('Sprungantwort des RC-Tiefpassfilters')
plt.xlabel('Zeit [s]')
plt.ylabel('Spannung [V]')
plt.grid(True, linestyle='--', alpha=0.7)
# Legende positionieren, um Plot nicht zu überlagern
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()