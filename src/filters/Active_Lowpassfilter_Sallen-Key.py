import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =================================================================
# 1. SCHALTUNGSPARAMETER & BERECHNUNGEN
# =================================================================

# --- Passiver RC-Tiefpassfilter (1. Ordnung) ---
R_passive = 1000       # Widerstand [Ohm]
C_passive = 1e-6       # Kapazität [Farad] (1 µF)
tau_passive = R_passive * C_passive  # Zeitkonstante τ = R*C [s]
fc_passive = 1 / (2 * np.pi * tau_passive)  # Eckfrequenz f_c [Hz]

# Übertragungsfunktion des passiven Filters: H_passive(s) = 1/(τ*s + 1)
num_passive = [1]                       # Zählerpolynom (Konstante)
den_passive = [tau_passive, 1]           # Nennerpolynom (τ*s + 1)

print("=== Passiver RC-Tiefpassfilter ===")
print(f"R = {R_passive} Ω, C = {C_passive:.6f} F")
print(f"τ = {tau_passive:.6f} s, f_c = {fc_passive:.2f} Hz\n")

# --- Aktiver Sallen-Key-Tiefpassfilter (2. Ordnung) ---
# Verwendet Operationsverstärker (hier: Einheitsverstärkung, symmetrische Komponenten)
R1 = R_passive     # Widerstand R1 [Ohm] (gleich passive R)
R2 = R_passive     # Widerstand R2 [Ohm] (gleich passive R)
C1 = C_passive     # Kapazität C1 [F] (gleich passive C)
C2 = C_passive     # Kapazität C2 [F] (gleich passive C)

# Parameter für den Nennerpolynom des aktiven Filters:
# H_active(s) = 1 / (A*s² + B*s + 1), mit A = R1*R2*C1*C2, B = R1*C1 + R2*C2
A = R1 * R2 * C1 * C2  # s²-Koeffizient
B = R1 * C1 + R2 * C2  # s-Koeffizient

# Übertragungsfunktion des aktiven Filters
num_active = [1]        # Zählerpolynom (Konstante)
den_active = [A, B, 1]  # Nennerpolynom (A*s² + B*s + 1)

# Eckfrequenz des aktiven Filters (entspricht passiver f_c)
fc_active = 1 / (2 * np.pi * np.sqrt(A))  # A = (R*C)^2 → sqrt(A) = R*C

print("=== Akuter Sallen-Key-Tiefpassfilter ===")
print(f"R1 = R2 = {R1} Ω, C1 = C2 = {C1:.6f} F")
print(f"A = {A:.10f}, B = {B:.6f}")
print(f"f_c (aktiv) = {fc_active:.2f} Hz (entspricht passiv)\n")

# =================================================================
# 2. BODE-DIAGRAMM (FREQUENZBEREICH)
# =================================================================
# Winkelfrequenzvektor (logarithmisch, 0.1 rad/s bis 1e5 rad/s)
w_rad = np.logspace(-1, 5, 500)  # ω in rad/s

# --- Frequenzgang des passiven Filters ---
w_p, H_passive = signal.freqs(num_passive, den_passive, w_rad)  # H(jω)
f_hz_p = w_p / (2 * np.pi)  # Frequenz f [Hz] (von ω umgerechnet)
gain_db_passive = 20 * np.log10(np.abs(H_passive))  # Gain in dB
phase_deg_passive = np.angle(H_passive, deg=True)    # Phase in °

# --- Frequenzgang des aktiven Filters ---
w_a, H_active = signal.freqs(num_active, den_active, w_rad)  # H(jω)
f_hz_a = w_a / (2 * np.pi)  # Frequenz f [Hz] (gleich wie passive, da w_rad identisch)
gain_db_active = 20 * np.log10(np.abs(H_active))  # Gain in dB
phase_deg_active = np.angle(H_active, deg=True)    # Phase in °

# =================================================================
# 3. PLOT: BODE-DIAGRAMM (Gain und Phase)
# =================================================================
plt.figure(figsize=(12, 8))

# --- Subplot 1: Amplitudengang (Gain) ---
plt.subplot(2, 1, 1)
plt.semilogx(f_hz_p, gain_db_passive, 'b-', linewidth=2, label='Passiv (1. Ordnung)')
plt.semilogx(f_hz_a, gain_db_active, 'r--', linewidth=2, label='Aktiv (Sallen-Key, 2. Ordnung)')
plt.title('Bode-Diagramm: Passiv RC vs. Aktiv Sallen-Key-Tiefpass')
plt.xlabel('Frequenz [Hz]')
plt.ylabel('Verstärkung [dB]')
plt.grid(which='both', linestyle='--', alpha=0.7)
# Eckfrequenz markieren
plt.axvline(fc_passive, color='g', linestyle='--', linewidth=1.5, 
            label=f'Eckfrequenz $f_c$ = {fc_passive:.2f} Hz')
plt.legend()
plt.ylim(-60, 10)  # Y-Achse begrenzen für bessere Sichtbarkeit
plt.xlim(f_hz_p.min(), f_hz_p.max())

# --- Subplot 2: Phasengang ---
plt.subplot(2, 1, 2)
plt.semilogx(f_hz_p, phase_deg_passive, 'b-', linewidth=2, label='Passiv (1. Ordnung)')
plt.semilogx(f_hz_a, phase_deg_active, 'r--', linewidth=2, label='Aktiv (2. Ordnung)')
plt.title('Phasengang')
plt.xlabel('Frequenz [Hz]')
plt.ylabel('Phase [°]')
plt.grid(which='both', linestyle='--', alpha=0.7)
# Eckfrequenz markieren
plt.axvline(fc_passive, color='g', linestyle='--', linewidth=1.5, label=f'$f_c$ = {fc_passive:.2f} Hz')
plt.legend()
plt.ylim(-180, 0)  # Phase sinkt bis -180° für 2. Ordnung
plt.xlim(f_hz_p.min(), f_hz_p.max())

plt.tight_layout()
plt.show()

# =================================================================
# 4. ZEITBEREICHS-SIMULATION: SPRUNGANTWORT
# =================================================================
# Zeitvektor (0 bis 5*τ, 500 Punkte)
t = np.linspace(0, 5 * tau_passive, 500)

# --- Simulierte Sprungantwort (passiv) ---
t_step_passive, y_step_passive = signal.step((num_passive, den_passive), T=t)
# Theoretische Sprungantwort (passiv): y(t) = 1 - e^(-t/τ)
y_theo_passive = 1 - np.exp(-t_step_passive / tau_passive)

# --- Simulierte Sprungantwort (aktiv) ---
t_step_active, y_step_active = signal.step((num_active, den_active), T=t)
# Theoretische Sprungantwort (aktiv): y(t) = 1 - e^(-t/τ) * (1 + t/τ) (abgeleitet aus H(s) = 1/(τ*s + 1)^2)
y_theo_active = 1 - np.exp(-t_step_active / tau_passive) * (1 + t_step_active / tau_passive)

# =================================================================
# 5. PLOT: SPRUNGANTWORT (Passiv vs. Aktiv)
# =================================================================
plt.figure(figsize=(10, 6))
plt.plot(t_step_passive, y_step_passive, 'b-', linewidth=2, label='Simulierte Passiv')
plt.plot(t_step_passive, y_theo_passive, 'b--', linewidth=1.5, label='Theoretische Passiv')
plt.plot(t_step_active, y_step_active, 'r-', linewidth=2, label='Simulierte Aktiv')
plt.plot(t_step_active, y_theo_active, 'r--', linewidth=1.5, label='Theoretische Aktiv')
# Eingangssprung (Referenz, 1V)
plt.plot(t, np.ones_like(t), 'k--', linewidth=1.5, alpha=0.6, label='$V_{in}(t)$ (Eingang)')
plt.title('Sprungantwort: Passiv RC vs. Aktiv Sallen-Key-Tiefpass')
plt.xlabel('Zeit [s]')
plt.ylabel('Spannung [V]')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(0, t.max())
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()
// Small rework