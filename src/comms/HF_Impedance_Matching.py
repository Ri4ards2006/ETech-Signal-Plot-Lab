"""
HF Impedance Matching Simulation & Visualization Toolkit - VERSION 1 (BASIC)

Kernfunktionalität: Simulation der Antennen-Impedanz und Berechnung des Return Loss 
über einen Frequenzbereich ohne Matching-Netzwerke.

Diese Version dient als Minimal Viable Product (MVP), um die numerische Basis 
(Impedanz, Gamma, Return Loss) zu etablieren.

1st version 
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 1. CORE IMPEDANCE FUNCTION ---

def compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0):
    """
    Berechnet die Antennen-Eingangsimpedanz (Z_ant) basierend auf dem RLC-Modell.
    
    Z_ant = R_ant + j*(X0 + ω*L_ant - 1/(ω*C_ant)).
    """
    if R_ant <= 0:
        R_ant = 1e-3  # Stabilitäts-Fallback
        
    omega = 2 * np.pi * f
    X_L = omega * L_ant
    
    # Vereinfachte Behandlung von C_ant (V1 lässt 1/(ωC) weg, falls C nicht vorhanden)
    X_C = -1/(omega * C_ant) if C_ant > 0 else 0
    
    X_ant = X0 + X_L + X_C
    return R_ant + 1j * X_ant

# --- 2. CORE MATCHING METRICS ---

def compute_gamma(Z_total, Z0):
    """Berechnet den Reflexionskoeffizienten Γ = (Z_total/Z0 - 1) / (Z_total/Z0 + 1)."""
    Z_norm = Z_total / Z0
    return (Z_norm - 1) / (Z_norm + 1)

def compute_return_loss(gamma):
    """Konvertiert den Reflexionskoeffizienten Γ in Return Loss (RL) in dB: RL = 20 * log10(|Γ|)."""
    # Verwendet np.clip, um -Inf-Werte (bei |Γ|=0) für die Darstellung zu vermeiden (optional, aber gut)
    gamma_magnitude = np.abs(gamma)
    gamma_magnitude[gamma_magnitude == 0] = 1e-12 
    return 20 * np.log10(gamma_magnitude)

# --- 3. BASIC SIMULATION AND PLOT FUNCTION (Non-Interactive) ---

def run_basic_simulation(f_min_mhz=5, f_max_mhz=20, R_ant=50, L_ant=1e-6, C_ant=1e-9, Z0=50):
    """Führt die Basissimulation der Antennenimpedanz und des Return Loss durch."""
    
    # Parameter Setup
    num_f = 500
    f_min_hz = f_min_mhz * 1e6
    f_max_hz = f_max_mhz * 1e6
    f = np.linspace(f_min_hz, f_max_hz, num_f)
    
    # Compute
    Z_ant = compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0=0) # X0 auf 0 gesetzt
    gamma_ant = compute_gamma(Z_ant, Z0)
    return_loss_ant = compute_return_loss(gamma_ant)
    
    # Plotting (Nur Return Loss)
    plt.figure(figsize=(10, 6))
    plt.plot(f/1e6, return_loss_ant, color='blue', linewidth=2, label='Antenna Return Loss (RL)')
    
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Return Loss (dB)')
    plt.title('V1: Return Loss Spectrum (Antenna Only)')
    plt.ylim(-60, 0) # Standard RL-Bereich
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()

# --- RUN V1 ---
if __name__ == '__main__':
    run_basic_simulation()