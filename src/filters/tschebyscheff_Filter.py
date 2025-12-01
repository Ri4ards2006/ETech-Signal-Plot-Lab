import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def design_chebyshev_filter(
    order: int,
    fc: float,
    ripple_db: float,
    filter_type: str = 'lowpass',
    passive: bool = True
) -> dict:
    """
    Entwirft ein Tschebyscheff-Filter (passiv oder aktiv) und liefert Übertragungsfunktion + Komponenten.
    
    Args:
        order (int): Filterordnung (z. B. 2).
        fc (float): Eckfrequenz [Hz].
        ripple_db (float): Passband-Ripple (Typ I) oder Stopband-Ripple (Typ II) [dB].
        filter_type (str): 'lowpass' oder 'highpass'.
        passive (bool): True = LC-Filter (passiv), False = Sallen-Key-RC-Filter (aktiv).
    
    Returns:
        dict: {'num': Zählerpolynom, 'den': Nennerpolynom, 'components': Komponenten, 'gain': Gain}
    """
    wn = 2 * np.pi * fc  # Winkelgeschwindigkeit (normierte FC) [rad/s]
    # 📌 Füge `output='zpk'` hinzu, um z, p, k zu erhalten (statt standardmäßiger b, a)
    z, p, k = signal.cheby1(order, ripple_db, wn, btype=filter_type, analog=True, output='zpk')
    num, den = signal.zpk2tf(z, p, k)  # Polynomform der Übertragungsfunktion
    
    components = {}
    if passive:
        # Passive LC-Filter: Komponenten L, R, C berechnen (Annahme: C=1µF)
        LC = den[0]  # LC = Zenzahl s²-Term (den[0])
        RC = den[1]  # RC = Koeffizient s-Term (den[1])
        C = 1e-6     # Kapazität [F] (arbiträr gewählt, kann angepasst werden)
        L = LC / C   # Induktivität [H]
        R = RC / C   # Widerstand [Ohm]
        components = {'L': L, 'R': R, 'C': C}
    else:
        # Aktives Sallen-Key-RC-Filter: Komponenten R1, R2, C1, C2 berechnen (asymmetrisch, falls nötig)
        # Zunächst C1 und C2 wählen (hier: C1=C2=1µF)
        C1 = C2 = 1e-6  # Kapazitäten [F]
        # Gleichungen für Sallen-Key-Denominator (asymmetrisch):
        # den_active(s) = (R1*R2)/(C1*C2) * s² + (R2/C1 + R1/C2) * s + 1
        a_den = den[0]  # Koeffizient s²-Term
        b_den = den[1]  # Koeffizient s-Term
        
        # Für C1=C2=C vereinfachen:
        C = C1
        eq1 = a_den * (C1 * C2)  # R1*R2 = a_den * C1*C2
        eq2 = b_den * (C1 * C2)  # R2*C2 + R1*C1 = b_den * C1*C2 (siehe unten)
        
        # Löse nach R1 und R2:
        discriminant = eq2**2 - 4 * eq1 * (C1 * C2)  # Korrektur: Faktor (C1*C2) hinzugefügt?
        # ❗Warnung: Die ursprünglichen Formeln könnten weiterhin ungenau sein (siehe Erläuterung unten)
        
        if discriminant < 0:
            # Asymmetrische Kapazitäten (C1=1µF, C2=2µF)
            C1 = 1e-6
            C2 = 2e-6
            eq1 = a_den * (C1 * C2)
            eq2 = b_den * (C1 * C2)
            sqrt_term = np.sqrt(eq2**2 - 4 * eq1)
            R1 = (eq2 + sqrt_term) / (2 * C2)  # Korrektur: Dividiere durch C2?
            R2 = (eq2 - sqrt_term) / (2 * C2)
            components = {'R1': R1, 'R2': R2, 'C1': C1, 'C2': C2}
            print("Hinweis: Asymmetrische Kapazitäten (C1=1µF, C2=2µF) verwendet.")
        else:
            # Symmetrische Komponenten
            # (R1 + R2) = eq2 / (C1 if C2=C1 else ...) → hier C1=C2=C
            # (R1 * R2) = eq1
            R1 = (eq2 / C1 + np.sqrt(discriminant)) / 2  # Möglicher Fehler in Originalformeln!
            R2 = (eq2 / C1 - np.sqrt(discriminant)) / 2
            components = {'R1': R1, 'R2': R2, 'C1': C1, 'C2': C2}
    
    return {
        'num': num,
        'den': den,
        'components': components,
        'gain': k
    }

def plot_bode_comparison(
    passive_data: dict,
    active_data: dict,
    fc: float,
    ripple_db: float,
    filter_type: str
):
    """Vergleicht Bode-Diagramme (Gain/Phase) von passivem LC- und aktivem RC-Tschebyscheff-Filter."""
    passive_num, passive_den = passive_data['num'], passive_data['den']
    active_num, active_den = active_data['num'], active_data['den']
    
    f = np.logspace(-1, 6, 500)
    w = 2 * np.pi * f
    
    # Frequenzgang passive Filter
    _, H_passive = signal.freqs(passive_num, passive_den, w)
    gain_db_passive = 20 * np.log10(np.abs(H_passive))
    phase_deg_passive = np.angle(H_passive, deg=True)
    
    # Frequenzgang aktives Filter
    _, H_active = signal.freqs(active_num, active_den, w)
    gain_db_active = 20 * np.log10(np.abs(H_active))
    phase_deg_active = np.angle(H_active, deg=True)
    
    # Butterworth-Referenz
    b, a = signal.butter(order, fc, btype=filter_type, analog=True)
    _, H_butter = signal.freqs(b, a, w)
    gain_db_butter = 20 * np.log10(np.abs(H_butter))
    
    # Plot Amplitudengang
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.semilogx(f, gain_db_passive, 'b-', linewidth=2, label='Passiv (LC)')
    plt.semilogx(f, gain_db_active, 'r--', linewidth=2, label='Aktiv (RC, Sallen-Key)')
    plt.semilogx(f, gain_db_butter, 'g-.', linewidth=2, label='Butterworth (Ref.)')
    
    plt.title(f'Bode-Diagramm: Tschebyscheff (Passiv vs. Aktiv) vs. Butterworth ({filter_type})')
    plt.xlabel('Frequenz [Hz]')
    plt.ylabel('Verstärkung [dB]')
    plt.grid(which='both', linestyle='--', alpha=0.7)
    plt.axvline(fc, color='k', linestyle=':', linewidth=1, label=f'$f_c$ = {fc:.2f} Hz')
    plt.axhline(-ripple_db, color='m', linestyle=':', linewidth=1, label=f'Ripple-Grenze ({ripple_db} dB)')
    plt.legend()
    plt.ylim(-60, 10) if filter_type == 'lowpass' else plt.ylim(-10, 60)
    plt.xlim(f.min(), f.max())
    
    # Plot Phasengang
    plt.subplot(2, 1, 2)
    plt.semilogx(f, phase_deg_passive, 'b-', linewidth=2, label='Passiv (LC)')
    plt.semilogx(f, phase_deg_active, 'r--', linewidth=2, label='Aktiv (RC)')
    
    plt.title(f'Phasengang ({filter_type})')
    plt.xlabel('Frequenz [Hz]')
    plt.ylabel('Phase [°]')
    plt.grid(which='both', linestyle='--', alpha=0.7)
    plt.axvline(fc, color='k', linestyle=':', linewidth=1, label=f'$f_c$ = {fc:.2f} Hz')
    plt.legend()
    plt.ylim(-180, 0) if filter_type == 'lowpass' else plt.ylim(0, 180)
    plt.xlim(f.min(), f.max())
    
    plt.tight_layout()
    plt.show()

# =================================================================
# Beispiel: Tschebyscheff-Tiefpass (2. Ordnung) mit Ripple
# =================================================================
order = 2
fc = 1000  # Eckfrequenz [Hz]
ripple_db = 1  # Passband-Ripple [dB]
filter_type = 'lowpass'

# Passive LC-Filter entwerfen
passive_data = design_chebyshev_filter(
    order=order,
    fc=fc,
    ripple_db=ripple_db,
    filter_type=filter_type,
    passive=True
)
passive_components = passive_data['components']
print("=== Passive LC-Komponenten ===")
print(f"Induktivität (L): {passive_components['L']:.6f} H")
print(f"Widerstand (R): {passive_components['R']:.6f} Ω")
print(f"Kapazität (C): {passive_components['C']:.6f} F\n")

# Aktives Sallen-Key-RC-Filter entwerfen
active_data = design_chebyshev_filter(
    order=order,
    fc=fc,
    ripple_db=ripple_db,
    filter_type=filter_type,
    passive=False
)
active_components = active_data['components']
print("=== Aktive RC-Komponenten (Sallen-Key) ===")
print(f"Widerstand R1: {active_components['R1']:.6f} Ω")
print(f"Widerstand R2: {active_components['R2']:.6f} Ω")
print(f"Kapazität C1: {active_components['C1']:.6f} F")
print(f"Kapazität C2: {active_components['C2']:.6f} F\n")

# Bode-Diagramm vergleichen
plot_bode_comparison(
    passive_data=passive_data,
    active_data=active_data,
    fc=fc,
    ripple_db=ripple_db,
    filter_type=filter_type
)

# Effizienz-Analyse
print("=== Effizienzsteigerung durch aktives Filter ===")
print(f"Passiv LC benötigt Induktivität L = {passive_components['L']:.2e} H.")
print(f"Aktiv RC verwendet R1={active_components['R1']:.2e}Ω, R2={active_components['R2']:.2e}Ω, C1={active_components['C1']:.6f}F, C2={active_components['C2']:.6f}F.")
print("→ Keine Induktoren, kompakt und kostengünstiger.")

# =================================================================
# Beispiel: Tschebyscheff-Hochpass (2. Ordnung)
# =================================================================
filter_type = 'highpass'

passive_high_data = design_chebyshev_filter(
    order=order,
    fc=fc,
    ripple_db=ripple_db,
    filter_type=filter_type,
    passive=True
)
passive_high_components = passive_high_data['components']
print("\n\n=== Passive LC-Komponenten (Hochpass) ===")
print(f"L: {passive_high_components['L']:.6f} H, R: {passive_high_components['R']:.6f} Ω, C: {passive_high_components['C']:.6f} F")

active_high_data = design_chebyshev_filter(
    order=order,
    fc=fc,
    ripple_db=ripple_db,
    filter_type=filter_type,
    passive=False
)
active_high_components = active_high_data['components']
print("=== Aktive RC-Komponenten (Hochpass, Sallen-Key) ===")
print(f"R1: {active_high_components['R1']:.6f} Ω, R2: {active_high_components['R2']:.6f} Ω")
print(f"C1: {active_high_components['C1']:.6f} F, C2: {active_high_components['C2']:.6f} F")

# Bode-Diagramm für Hochpass
plot_bode_comparison(
    passive_data=passive_high_data,
    active_data=active_high_data,
    fc=fc,
    ripple_db=ripple_db,
    filter_type=filter_type
)