import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, Checkbox
from scipy import signal

# =================================================================
# 1. IMPEDANZ-BERECHNUNGEN & TRANSFORMATIONEN
# =================================================================
def compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0):
    """
    Calculate antenna input impedance Z_ant = R_ant + j*(X0 + ω*L_ant - 1/(ω*C_ant))
    Args:
        f (np.ndarray): Frequenz (Hz)
        R_ant (float): Antennenwiderstand (Ω)
        L_ant (float): Antennenserienduktivität (H)
        C_ant (float): Antennenparallelkapazität (F)
        X0 (float): Zusätzliche parasitäre Reaktanz (Ω)
    Returns:
        np.ndarray: Komplexe Antennenimpedanz (Ω)
    """
    omega = 2 * np.pi * f
    X_L = omega * L_ant
    X_C = -1/(omega * C_ant) if C_ant > 1e-15 else 0  # Vermeide Division durch 0
    return R_ant + 1j * (X0 + X_L + X_C)

def series_L_transform(Z_in, f, L_L):
    """
    Transformiere Impedanz mit Serieninduktivität (L) zur Kompensation kapazitiver Reaktion
    Args:
        Z_in (np.ndarray): Eingangsimpedanz (Ω, komplex)
        f (np.ndarray): Frequenz (Hz)
        L_L (float): Serieninduktivität (H)
    Returns:
        np.ndarray: Transformierte Impedanz (Ω)
    """
    omega = 2 * np.pi * f
    return Z_in + 1j * omega * L_L

def shunt_C_transform(Z_in, f, C_C):
    """
    Transformiere Impedanz mit Parallelkapazität (C) zur Kompensation inductiver Reaktion
    Args:
        Z_in (np.ndarray): Eingangsimpedanz (Ω, komplex)
        f (np.ndarray): Frequenz (Hz)
        C_C (float): Parallelkapazität (F)
    Returns:
        np.ndarray: Transformierte Impedanz (Ω)
    """
    if C_C <= 1e-15:  # Behandle 0/Fehler als offenes Netz
        return Z_in
    omega = 2 * np.pi * f
    Z_C = 1/(1j * omega * C_C)
    return (Z_in * Z_C) / (Z_in + Z_C)

def series_L_shunt_C_transform(Z_in, f, L_L, C_C):
    """
    Kombinierte Transformation (Serien L + Parallelkapazität C)
    Args:
        Z_in (np.ndarray): Eingangsimpedanz (Ω, komplex)
        f (np.ndarray): Frequenz (Hz)
        L_L (float): Serieninduktivität (H)
        C_C (float): Parallelkapazität (F)
    Returns:
        np.ndarray: Transformierte Impedanz (Ω)
    """
    return shunt_C_transform(series_L_transform(Z_in, f, L_L), f, C_C)

def shunt_L_series_C_transform(Z_in, f, L_L, C_C):
    """
    Transformation mit paralleler Induktivität (L) und seriener Kapazität (C)
    Args:
        Z_in (np.ndarray): Eingangsimpedanz (Ω, komplex)
        f (np.ndarray): Frequenz (Hz)
        L_L (float): Paralleler Induktivität (H)
        C_C (float): Seriener Kapazität (F)
    Returns:
        np.ndarray: Transformierte Impedanz (Ω)
    """
    if L_L <= 1e-15:  # Paralleler L ignoriert, falls ≤0
        return Z_in
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L
    Z_parallel = (Z_in * Z_L) / (Z_in + Z_L)  # Z_in || Z_L
    
    if C_C <= 1e-15:  # Seriener C ignoriert, falls ≤0
        return Z_parallel
    Z_C = 1/(1j * omega * C_C)
    return Z_parallel + Z_C  # Serie mit C

def pi_network_transform(Z_in, f, L1, L2, C):
    """
    π-Netzwerk-Transformation (L1 seriend, L2 seriend, C paralleler)
    Args:
        Z_in (np.ndarray): Eingangsimpedanz (Ω, komplex)
        f (np.ndarray): Frequenz (Hz)
        L1 (float): Erste Serieninduktivität (H)
        L2 (float): Zweite Serieninduktivität (H)
        C (float): Parallelkapazität (F)
    Returns:
        np.ndarray: Transformierte Impedanz (Ω)
    """
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1
    Z_L2 = 1j * omega * L2
    
    if C <= 1e-15:
        Z_C = np.inf  # Offenes Netz (kein C-Effekt)
    else:
        Z_C = 1/(1j * omega * C)
    
    # L2 || C
    Z_parallel = (Z_L2 * Z_C) / (Z_L2 + Z_C) if Z_C != np.inf else Z_L2
    return Z_in + Z_L1 + Z_parallel  # Z_in + L1 + (L2 || C)

def t_network_transform(Z_in, f, L1, L2, C):
    """
    T-Netzwerk-Transformation (L1 parallel, C seriend, L2 parallel)
    Args:
        Z_in (np.ndarray): Eingangsimpedanz (Ω, komplex)
        f (np.ndarray): Frequenz (Hz)
        L1 (float): Erste parallele Induktivität (H)
        L2 (float): Zweite parallele Induktivität (H)
        C (float): Seriener Kapazität (F)
    Returns:
        np.ndarray: Transformierte Impedanz (Ω)
    """
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1 if L1 > 1e-15 else np.inf
    Z_L2 = 1j * omega * L2 if L2 > 1e-15 else np.inf
    
    # Z_in || L1
    if Z_L1 != np.inf:
        Z_parallel1 = (Z_in * Z_L1) / (Z_in + Z_L1)
    else:
        Z_parallel1 = Z_in
    
    # + Serie C
    if C <= 1e-15:
        Z_series = Z_parallel1
    else:
        Z_C = 1/(1j * omega * C)
        Z_series = Z_parallel1 + Z_C
    
    # || L2
    if Z_L2 != np.inf:
        return (Z_series * Z_L2) / (Z_series + Z_L2)
    else:
        return Z_series

# =================================================================
# 2. REFLEKTION Koeffizient & Return Loss
# =================================================================
def compute_gamma(Z_total, Z0):
    """Berechne Reflexionskoeffizient Γ = (Z/Z0 - 1)/(Z/Z0 + 1)"""
    Z_norm = Z_total / Z0
    return (Z_norm - 1) / (Z_norm + 1)

def compute_return_loss(gamma):
    """Return Loss (dB) = -20*log10(|Γ|) (perfekter Match → RL→∞ dB)"""
    gamma_abs = np.abs(gamma)
    return_loss = np.where(gamma_abs == 0, np.inf, -20 * np.log10(gamma_abs))
    return np.clip(return_loss, -60, 100)  # Clip to avoid inf in plots

# =================================================================
# 3. INTERAKTIVE PLOT-FUNKTIONEN
# =================================================================
def plot_smith_chart(ax, Z0):
    """Smith-Chart mit Grid (Constant R/X) und Perfect-Match-Marker"""
    theta = np.linspace(0, 2*np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), color='k', linestyle='--', lw=1)  # Unit circle
    
    # Perfect Match (Γ=0) markieren
    ax.scatter(0, 0, color='g', s=100, zorder=5, marker='*')
    ax.text(0, 0, 'Perfect\nMatch', color='g', zorder=6, ha='right', va='center')
    
    # Constant R_norm Grid (subtler)
    R_norm_vals = np.concatenate([np.arange(0, 2, 0.5), np.arange(2.5, 10, 0.5), [10]])
    for R_norm in R_norm_vals:
        X_norm = np.linspace(-10, 10, 1000)
        Z_norm = R_norm + 1j * X_norm
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, color='gray', linestyle='-', lw=0.5, alpha=0.5)
    
    # Constant X_norm Grid (subtler)
    X_norm_vals = np.arange(-10, 10.1, 0.5)
    for X_norm in X_norm_vals:
        R_norm = np.linspace(0, 10, 1000)
        Z_norm = R_norm + 1j * X_norm
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, color='gray', linestyle='-', lw=0.5, alpha=0.5)
    
    ax.set_xlabel('Re(Γ)')
    ax.set_ylabel('Im(Γ)')
    ax.set_title(f'Smith Chart (Z0={Z0} Ω)')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(False)

def plot_return_loss(ax, f, gamma_total, gamma_ant, Z0):
    """Return-Loss-Spektrum mit Anmerkungen zum besten Match"""
    rl_total = compute_return_loss(gamma_total)
    rl_ant = compute_return_loss(gamma_ant)
    
    ax.plot(f/1e6, rl_total, color='r', lw=2, label='With Network')
    ax.plot(f/1e6, rl_ant, color='b', ls='--', lw=1.5, label='Without Network')
    
    # Best RL (Network)
    best_rl_idx = np.argmax(rl_total)
    best_rl = rl_total[best_rl_idx]
    f_best = f[best_rl_idx]/1e6
    ax.scatter(f_best, best_rl, color='g', s=80, zorder=3)
    ax.text(f_best, best_rl, f'Best RL: {best_rl:.1f} dB\nf={f_best:.2f} MHz', 
            color='g', va='bottom', ha='center', fontsize=10)
    
    # Best RL (Antenna)
    best_rl_ant_idx = np.argmax(rl_ant)
    best_rl_ant = rl_ant[best_rl_ant_idx]
    f_best_ant = f[best_rl_ant_idx]/1e6
    ax.scatter(f_best_ant, best_rl_ant, color='orange', s=80, zorder=3)
    ax.text(f_best_ant, best_rl_ant, f'Antenna RL: {best_rl_ant:.1f} dB\nf={f_best_ant:.2f} MHz', 
            color='orange', va='bottom', ha='center', fontsize=10)
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Return Loss (dB)')
    ax.set_title('Return Loss Spectrum')
    ax.set_ylim(-60, 80)  # Realistischer Bereich (schlechte Match: -60 dB, gute Match: +80 dB)
    ax.grid(ls='--', alpha=0.6)
    ax.legend()

def plot_normalized_impedance(ax, f, Z_total, Z_ant, Z0):
    """Plot von R/Z0 und X/Z0-Komponenten"""
    Z_total_norm = Z_total / Z0
    Z_ant_norm = Z_ant / Z0
    
    ax.plot(f/1e6, Z_ant_norm.real, 'b--', lw=1.5, label='Antenna R/Z0')
    ax.plot(f/1e6, Z_total_norm.real, 'g', lw=2, label='Total R/Z0')
    ax.plot(f/1e6, Z_ant_norm.imag, 'r--', lw=1.5, label='Antenna X/Z0')
    ax.plot(f/1e6, Z_total_norm.imag, 'purple', lw=2, label='Total X/Z0')
    
    ax.axhline(1, color='k', ls=':', lw=1)  # R/Z0=1 (perfekter Widerstand)
    ax.axhline(0, color='k', ls=':', lw=1)  # X/Z0=0 (keine Reaktion)
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized Impedance')
    ax.set_title('Normalized Impedance Components (R/Z0, X/Z0)')
    ax.grid(ls='--', alpha=0.6)
    ax.legend()

def update_plots(f_min, f_max, num_f_points, Z0,
                network_type,
                # L-Parameter (µH)
                L_L_series, L_C_shunt, L1_pi_t, L2_pi_t,
                # C-Parameter (nF)
                C_C_shunt, C_pi_t,
                R_ant, L_ant_muH, C_ant_nF, X0):
    """
    Hauptsimulation mit interaktiven Einstellungen
    Args:
        f_min (float): Startfrequenz (MHz)
        f_max (float): Endfrequenz (MHz)
        num_f_points (int): Anzahl Frequenzpunkte
        Z0 (float): Referenzimpedanz (Ω)
        network_type (str): Netzwerk-Typ
        L_L_series (float): Serieninduktivität (µH)
        L_C_shunt (float): Paralleler Induktivität (µH) (nur für Shunt L + Series C)
        L1_pi_t (float): L1 für Pi/T-Netzwerk (µH)
        L2_pi_t (float): L2 für Pi/T-Netzwerk (µH)
        C_C_shunt (float): Parallelkapazität (nF) (nur für C-Netzwerk/Series L+Shunt C)
        C_pi_t (float): Kapazität für Pi/T-Netzwerk (nF)
        R_ant (float): Antennenwiderstand (Ω)
        L_ant_muH (float): Antennenserienduktivität (µH)
        C_ant_nF (float): Antennenparallelkapazität (nF)
        X0 (float): Parasitäre Reaktanz (Ω)
    """
    # Parameterkonvertierung (µH/nF → H/F)
    L_L_series_H = L_L_series * 1e-6
    L_C_shunt_H = L_C_shunt * 1e-6
    L1_pi_t_H = L1_pi_t * 1e-6
    L2_pi_t_H = L2_pi_t * 1e-6
    C_C_shunt_F = C_C_shunt * 1e-9
    C_pi_t_F = C_pi_t * 1e-9
    L_ant_H = L_ant_muH * 1e-6
    C_ant_F = C_ant_nF * 1e-9

    # Frequenzvektor generieren
    f_hz = np.linspace(f_min * 1e6, f_max * 1e6, num_f_points)

    # Roh-Antennenimpedanz (ohne Netzwerk)
    Z_ant = compute_antenna_impedance(f_hz, R_ant, L_ant_H, C_ant_F, X0)
    gamma_ant = compute_gamma(Z_ant, Z0)

    # Transformierte Impedanz (mit Netzwerk)
    if network_type == 'None':
        Z_total = Z_ant
    elif network_type == 'L Network (Series L)':
        Z_total = series_L_transform(Z_ant, f_hz, L_L_series_H)
    elif network_type == 'C Network (Shunt C)':
        Z_total = shunt_C_transform(Z_ant, f_hz, C_C_shunt_F)
    elif network_type == 'Series L + Shunt C':
        Z_total = series_L_shunt_C_transform(Z_ant, f_hz, L_L_series_H, C_C_shunt_F)
    elif network_type == 'Shunt L + Series C':
        Z_total = shunt_L_series_C_transform(Z_ant, f_hz, L_C_shunt_H, C_pi_t_F)  # C_pi_t hier verwendet (passend für Shunt L + Series C)
    elif network_type == 'Pi Network (L1, L2, C)':
        Z_total = pi_network_transform(Z_ant, f_hz, L1_pi_t_H, L2_pi_t_H, C_pi_t_F)
    elif network_type == 'T Network (L1, L2, C)':
        Z_total = t_network_transform(Z_ant, f_hz, L1_pi_t_H, L2_pi_t_H, C_pi_t_F)
    else:
        Z_total = Z_ant

    # Reflexionskoeffizient nach Netzwerk
    gamma_total = compute_gamma(Z_total, Z0)

    # Plot-Figur initialisieren
    fig, (ax_smith, ax_rl, ax_z) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Smith-Chart
    plot_smith_chart(ax_smith, Z0)
    ax_smith.plot(gamma_ant.real, gamma_ant.imag, 'b--', lw=1.5, label='Antenna (No Match)')
    ax_smith.plot(gamma_total.real, gamma_total.imag, 'r', lw=2, label=f'Network: {network_type}')
    ax_smith.legend(fontsize=10)

    # Return-Loss-Spektrum
    plot_return_loss(ax_rl, f_hz, gamma_total, gamma_ant, Z0)

    # Normalisierte Impedanz
    plot_normalized_impedance(ax_z, f_hz, Z_total, Z_ant, Z0)

    plt.tight_layout()
    plt.show()

# =================================================================
# 4. INTERAKTIVE WIDGET-EINSTELLUNGEN
# =================================================================
interact(
    update_plots,
    
    # Frequenzbereich
    f_min=FloatSlider(min=1, max=100, step=0.5, value=5, description='Min Frequenz (MHz)'),
    f_max=FloatSlider(min=1, max=100, step=0.5, value=20, description='Max Frequenz (MHz)'),
    num_f_points=IntSlider(min=100, max=2000, step=100, value=1000, description='Frequenzpunkte'),
    
    # Referenzimpedanz
    Z0=FloatSlider(min=25, max=100, step=5, value=50, description='Reference Z0 (Ω)'),
    
    # Netzwerk-Typ
    network_type=Dropdown(
        options=['None', 'L Network (Series L)', 'C Network (Shunt C)',
                 'Series L + Shunt C', 'Shunt L + Series C',
                 'Pi Network (L1, L2, C)', 'T Network (L1, L2, C)'],
        value='None',
        description='Netzwerk-Typ:'
    ),
    
    # L-Parameter (µH)
    L_L_series=FloatSlider(min=0, max=100, step=0.1, value=1, description='Series L (µH)'),
    L_C_shunt=FloatSlider(min=0, max=100, step=0.1, value=1, description='Shunt L (µH)'),  # Für Shunt L + Series C
    L1_pi_t=FloatSlider(min=0, max=100, step=0.1, value=1, description='Pi/T L1 (µH)'),
    L2_pi_t=FloatSlider(min=0, max=100, step=0.1, value=1, description='Pi/T L2 (µH)'),
    
    # C-Parameter (nF)
    C_C_shunt=FloatSlider(min=0, max=100, step=0.1, value=1, description='Shunt C (nF)'),
    C_pi_t=FloatSlider(min=0, max=100, step=0.1, value=1, description='Pi/T C (nF)'),
    
    # Antennenparameter
    R_ant=FloatSlider(min=10, max=100, step=5, value=50, description='Antennen R (Ω)'),
    L_ant_muH=FloatSlider(min=0, max=100, step=0.1, value=1, description='Antennen L (µH)'),
    C_ant_nF=FloatSlider(min=0, max=100, step=0.1, value=1, description='Antennen C (nF)'),
    X0=FloatSlider(min=-100, max=100, step=1, value=0, description='Parasitäre X0 (Ω)')
);