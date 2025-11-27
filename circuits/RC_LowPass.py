import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =================================================================
# 1. SCHALTUNGSPARAMETER & BERECHNUNGEN
# =================================================================

# Schaltung: RC-Tiefpassfilter (Ausgang über C)
R = 1000  # Widerstand in Ohm
C = 1e-6  # Kapazität in Farad (1 µF)
tau = R * C # Zeitkonstante
fc = 1 / (2 * np.pi * tau)  # Eckfrequenz/Grenzfrequenz in Hz

print(f"Schaltung: RC-Tiefpassfilter")
print(f"Zeitkonstante (tau): {tau:.6f} s")
print(f"Eckfrequenz (fc): {fc:.2f} Hz\n")

# Koeffizienten der Übertragungsfunktion H(s) = 1 / (1 + s*tau)
# Zähler-Koeffizienten (Numerator)
num = [1]
# Nenner-Koeffizienten (Denominator)
den = [tau, 1] 

# =================================================================
# 2. BODE-DIAGRAMM (FREQUENZBEREICH)
# =================================================================

# Erstellen eines Vektors von Winkelfrequenzen (logarithmisch)
w = np.logspace(-1, 5, 500) 

# Berechnen des Frequenzgangs H(j*w)
w_rad, H = signal.freqs(num, den, w)

# Umrechnung von Winkelfrequenz (rad/s) in Frequenz (Hz)
f_hz = w_rad / (2 * np.pi)

# Berechnen von Verstärkung (Gain) in dB und Phase in Grad
gain_db = 20 * np.log10(abs(H))
phase_deg = np.angle(H, deg=True)

plt.figure(figsize=(10, 8))

# --- Subplot 1: Verstärkung (Gain) ---
plt.subplot(2, 1, 1)
plt.semilogx(f_hz, gain_db)
plt.title('Bode Diagramm des RC-Tiefpassfilters')
plt.ylabel('Verstärkung [dB]')
plt.grid(which='both', axis='both')
plt.axvline(fc, color='r', linestyle='--', label=f'Eckfrequenz $f_c$ = {fc:.2f} Hz')
plt.legend()

# --- Subplot 2: Phase ---
plt.subplot(2, 1, 2)
plt.semilogx(f_hz, phase_deg)
plt.xlabel('Frequenz [Hz]')
plt.ylabel('Phase [Grad]')
plt.grid(which='both', axis='both')
plt.show()

# =================================================================
# 3. ZEITBEREICHS-SIMULATION (SPRUNGANTWORT)
# =================================================================

# Erstellen eines Zeitvektors (von 0 bis 5*tau, um das Einschwingen zu sehen)
t = np.linspace(0, 5 * tau, 500)

# Berechnen der Sprungantwort (Antwort auf eine Eingangsspannung von 1 V)
t_step, y_step = signal.step(num, den, T=t)

plt.figure(figsize=(8, 4))
plt.plot(t_step, y_step, label='Gefilterte Ausgangsspannung $V_{out}$')
plt.plot(t_step, np.ones_like(t_step), 'k--', alpha=0.5, label='Eingangsspannung $V_{in}$ (Sprung)')
plt.title('Sprungantwort des RC-Tiefpassfilters (Zeitbereich)')
plt.xlabel('Zeit [s]')
plt.ylabel('Spannung [V]')
plt.grid(True)
plt.legend()
plt.show()