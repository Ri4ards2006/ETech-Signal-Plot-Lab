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

# Koeffizienten der Übertragungsfunktion H(s) = 1 / (1 + τ*s) 
# (Zähler: [1], Nenner: [τ, 1] → Polynom: 1 + τ*s)
num = [1]
den = [tau, 1]  # KOORREKTE NENNERKOEFFIZIENTEN (ursprünglich korrekt, vorherige Korrektur war fehlerhaft)

# =================================================================
# 2. BODE-DIAGRAMM (FREQUENZBEREICH)
# =================================================================
# Winkelfrequenzvektor (logarithmisch, von 0.1 rad/s bis 1e5 rad/s)
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
plt.semilogx(f_hz, gain_db, linewidth=2)  # Linienstärke erhöhen
plt.title('Bode-Diagramm des RC-Tiefpassfilters', fontsize=12, pad=15)
plt.xlabel('Frequenz [Hz]', fontsize=10)
plt.ylabel('Verstärkung [dB]', fontsize=10)
plt.grid(which='both', axis='both', linestyle='--', alpha=0.6)  # Grid deutlicher
# Eckfrequenz markieren (Genauigkeit: -3 dB)
plt.axvline(fc, color='r', linestyle='--', linewidth=2, 
            label=f'Eckfrequenz $f_c$ = {fc:.2f} Hz (Gain: -3 dB)')
plt.legend(loc='upper right', fontsize=10)
plt.ylim(top=5, bottom=-60)  # Gain-Achse begrenzen (Typisch für Tiefpass)
plt.xlim(left=f_hz.min(), right=f_hz.max())  # Frequenz-Achse begrenzen

# --- Subplot 2: Phase ---
plt.subplot(2, 1, 2)
plt.semilogx(f_hz, phase_deg, linewidth=2)
plt.title('Phasengang', fontsize=12, pad=15)
plt.xlabel('Frequenz [Hz]', fontsize=10)
plt.ylabel('Phase [°]', fontsize=10)
plt.grid(which='both', axis='both', linestyle='--', alpha=0.6)
plt.axvline(fc, color='r', linestyle='--', linewidth=2, label=f'$f_c$ = {fc:.2f} Hz')
plt.legend(loc='lower right', fontsize=10)
plt.ylim(-90, 0)  # Phase zwischen -90° und 0° (Typisch für Tiefpass)
plt.xlim(left=f_hz.min(), right=f_hz.max())  # Frequenz-Achse begrenzen

plt.tight_layout()  # Optimales Layout für Subplots
plt.show()

# =================================================================
# 3. ZEITBEREICHS-SIMULATION (SPRUNGANTWORT)
# =================================================================
# Zeitvektor (0 bis 5*τ, 500 Punkte → detaillierter Plot)
t = np.linspace(0, 5 * tau, 500)

# Sprungantwort berechnen (System als Tupel (num, den) übergeben)
t_step, y_step = signal.step((num, den), T=t)  

# Theoretische Sprungantwort (Validierung)
y_theo = 1 - np.exp(-t_step / tau)

# Plotte Sprungantwort
plt.figure(figsize=(8, 4))
plt.plot(t_step, y_step, 'b-', linewidth=2, label='Simulierte $V_{out}(t)$')
plt.plot(t_step, y_theo, 'r--', linewidth=1.5, label='Theoretische $V_{out}(t)$')
plt.plot(t_step, np.ones_like(t_step), 'k--', linewidth=1.5, alpha=0.6, 
         label='$V_{in}(t)$ (Eingangssprung)')
plt.title('Sprungantwort des RC-Tiefpassfilters', fontsize=12, pad=15)
plt.xlabel('Zeit [s]', fontsize=10)
plt.ylabel('Spannung [V]', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)
plt.xlim(0, 5 * tau)  # Zeitachse begrenzen (keine leeren Bereiche)
plt.ylim(0, 1.1)  # Spannungsbereich etwas übersichtlicher
plt.tight_layout()
plt.show() 
// small reworkss