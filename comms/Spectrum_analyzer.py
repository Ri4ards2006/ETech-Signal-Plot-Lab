import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown

# =================================================================
# 1. Impedanz-Berechnungen & Transformationen
# =================================================================
def compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0):
    """
    Berechne die Eingangsimpedanz einer Antenne (ohne Netzwerk)
    Args:
        f (np.ndarray): Frequenz (Hz)
        R_ant (float): Antennenwiderstand (Ω)
        L_ant (float): Serienduktivität der Antenne (H)
        C_ant (float): Parallelkapazität der Antenne (F)
        X0 (float): Zusätzliche parasitäre Reaktanz (Ω)
    Returns:
        np.ndarray: Komplexe Impedanz Z_ant (Ω)
    """
    omega = 2 * np.pi * f
    X_L = omega * L_ant
    X_C = -1/(omega * C_ant) if C_ant > 1e-15 else 0  # Vermeide Division durch 0
    return R_ant + 1j * (X0 + X_L + X_C)

def series_L_transform(Z_in, f, L_L):
    """Transformiere Impedanz mit Serieninduktivität (L)"""
    omega = 2 * np.pi * f
    return Z_in + 1j * omega * L_L

def shunt_C_transform(Z_in, f, C_C):
    """Transformiere Impedanz mit Parallelkapazität (C)"""
    if C_C <= 1e-15:
        return Z_in  # Kein Effekt, wenn C ≈ 0
    omega = 2 * np.pi * f
    Z_C = 1/(1j * omega * C_C)
    return (Z_in * Z_C) / (Z_in + Z_C)  # Z_in || Z_C

def series_L_shunt_C_transform(Z_in, f, L_L, C_C):
    """Kombinierte Transformation: Serien L + Parallel C"""
    Z_series_L = series_L_transform(Z_in, f, L_L)
    return shunt_C_transform(Z_series_L, f, C_C)

def shunt_L_series_C_transform(Z_in, f, L_L, C_C):
    """Transformation: Paralleler L + Seriener C"""
    if L_L <= 1e-15:
        return Z_in  # Paralleler L ignoriert
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L
    Z_parallel_L = (Z_in * Z_L) / (Z_in + Z_L)  # Z_in || L_L
    
    if C_C <= 1e-15:
        return Z_parallel_L  # Seriener C ignoriert
    Z_C = 1/(1j * omega * C_C)
    return Z_parallel_L + Z_C  # Serie mit C

def pi_network_transform(Z_in, f, L1, L2, C):
    """π-Netzwerk-Transformation (L1, L2 seriend; C parallel)"""
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1
    Z_L2 = 1j * omega * L2
    
    # Kapazität C in Ohm konvertieren (oder unendlich, wenn C ≈ 0)
    if C <= 1e-15:
        Z_C = np.inf
    else:
        Z_C = 1/(1j * omega * C)
    
    # L2 || C
    Z_parallel_L2C = (Z_L2 * Z_C) / (Z_L2 + Z_C) if Z_C != np.inf else Z_L2
    return Z_in + Z_L1 + Z_parallel_L2C  # Z_in + L1 + (L2 || C)

def t_network_transform(Z_in, f, L1, L2, C):
    """T-Netzwerk-Transformation (L1 parallel, C seriend, L2 parallel)"""
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1 if L1 > 1e-15 else np.inf
    Z_L2 = 1j * omega * L2 if L2 > 1e-15 else np.inf
    
    # Z_in || L1
    Z_parallel1 = (Z_in * Z_L1) / (Z_in + Z_L1) if Z_L1 != np.inf else Z_in
    
    # + Serie C
    if C <= 1e-15:
        Z_series = Z_parallel1
    else:
        Z_C = 1/(1j * omega * C)
        Z_series = Z_parallel1 + Z_C
    
    # || L2
    Z_total = (Z_series * Z_L2) / (Z_series + Z_L2) if Z_L2 != np.inf else Z_series
    return Z_total

# =================================================================
# 2. Reflexionskoeffizient & Return Loss
# =================================================================
def compute_gamma(Z_total, Z0):
    """Rechnet Reflexionskoeffizient Γ = (Z/Z0 - 1)/(Z/Z0 + 1)"""
    Z_norm = Z_total / Z0
    return (Z_norm - 1) / (Z_norm + 1)

def compute_return_loss(gamma):
    """Konvertiert Γ zu Return Loss (dB), clippt extreme Werte."""
    gamma_abs = np.abs(gamma)
    return_loss = np.where(gamma_abs == 0, np.inf, -20 * np.log10(gamma_abs))
    return np.clip(return_loss, -60, 100)  # Verhindert Inf in Plots

# =================================================================
# 3. Interaktive Plot-Funktionen
# =================================================================
def plot_smith_chart(ax, Z0):
    """Smith-Chart mit Grid und Perfect-Match-Marker"""
    theta = np.linspace(0, 2*np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)  # Einheitskreis
    
    # Perfect Match (Γ=0)
    ax.scatter(0, 0, c='g', s=100, zorder=5, marker='*')
    ax.text(0, 0, 'Perfect\nMatch', c='g', zorder=6, ha='right', va='center')
    
    # Konstante R_norm-Grid (subtler)
    R_norm_vals = np.concatenate([np.arange(0, 2, 0.5), np.arange(2.5, 10, 0.5), [10]])
    for R_norm in R_norm_vals:
        X_norm = np.linspace(-10, 10, 1000)
        Z_norm = R_norm + 1j*X_norm
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, 'gray', ls='-', lw=0.5, alpha=0.5)
    
    # Konstante X_norm-Grid (subtler)
    X_norm_vals = np.arange(-10, 10.1, 0.5)
    for X_norm in X_norm_vals:
        R_norm = np.linspace(0, 10, 1000)
        Z_norm = R_norm + 1j*X_norm
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, 'gray', ls='-', lw=0.5, alpha=0.5)
    
    ax.set_xlabel('Re(Γ)'), ax.set_ylabel('Im(Γ)')
    ax.set_title(f'Smith Chart (Z0={Z0} Ω)'), ax.grid(False)
    ax.set_xlim(-1.2, 1.2), ax.set_ylim(-1.2, 1.2), ax.set_aspect('equal')

def plot_return_loss(ax, f, gamma_total, gamma_ant, Z0):
    """Return-Loss-Spektrum mit Anmerkungen zum besten Match"""
    rl_total = compute_return_loss(gamma_total)
    rl_ant = compute_return_loss(gamma_ant)
    
    ax.plot(f/1e6, rl_total, 'r', lw=2, label='Mit Netzwerk')
    ax.plot(f/1e6, rl_ant, 'b--', lw=1.5, label='Ohne Netzwerk')
    
    # Bestes RL (mit Netzwerk)
    best_idx = np.argmax(rl_total)
    ax.scatter(f[best_idx]/1e6, rl_total[best_idx], c='g', s=80, zorder=3)
    ax.text(f[best_idx]/1e6, rl_total[best_idx], 
            f'Best RL: {rl_total[best_idx]:.1f} dB\nf={f[best_idx]/1e6:.2f} MHz', 
            c='g', va='bottom', ha='center', fontsize=10)
    
    # Bestes RL (Antenne allein)
    best_idx_ant = np.argmax(rl_ant)
    ax.scatter(f[best_idx_ant]/1e6, rl_ant[best_idx_ant], c='orange', s=80, zorder=3)
    ax.text(f[best_idx_ant]/1e6, rl_ant[best_idx_ant], 
            f'Antenne RL: {rl_ant[best_idx_ant]:.1f} dB\nf={f[best_idx_ant]/1e6:.2f} MHz', 
            c='orange', va='bottom', ha='center', fontsize=10)
    
    ax.set_xlabel('Frequenz (MHz)'), ax.set_ylabel('Return Loss (dB)')
    ax.set_title('Return Loss Spectrum'), ax.set_ylim(-60, 80)
    ax.grid(ls='--', alpha=0.6), ax.legend()

def plot_normalized_impedance(ax, f, Z_total, Z_ant, Z0):
    """Plot von R/Z0 und X/Z0-Komponenten"""
    Z_total_norm = Z_total / Z0
    Z_ant_norm = Z_ant / Z0
    
    ax.plot(f/1e6, Z_ant_norm.real, 'b--', lw=1.5, label='Antenne R/Z0')
    ax.plot(f/1e6, Z_total_norm.real, 'g', lw=2, label='Gesamt R/Z0')
    ax.plot(f/1e6, Z_ant_norm.imag, 'r--', lw=1.5, label='Antenne X/Z0')
    ax.plot(f/1e6, Z_total_norm.imag, 'purple', lw=2, label='Gesamt X/Z0')
    
    ax.axhline(1, c='k', ls=':', lw=1, label='Perfect R/Z0')  # R/Z0=1
    ax.axhline(0, c='k', ls=':', lw=1, label='Perfect X/Z0')  # X/Z0=0
    ax.set_xlabel('Frequenz (MHz)'), ax.set_ylabel('Normierte Impedanz')
    ax.set_title('Normierte Impedanz-Komponenten (R/Z0, X/Z0)')
    ax.grid(ls='--', alpha=0.6), ax.legend()

def update_plots(f_min, f_max, num_f_points, Z0,
                network_type,
                L_L_series, L_C_shunt, L1_pi_t, L2_pi_t,
                C_C_shunt, C_pi_t,
                R_ant, L_ant_muH, C_ant_nF, X0):
    """Hauptfunktion zur Simulation und interaktiven Visualisierung"""
    # Parameterkonvertierung (µH/nF → H/F)
    L_series = L_L_series * 1e-6
    L_shunt = L_C_shunt * 1e-6
    L1 = L1_pi_t * 1e-6
    L2 = L2_pi_t * 1e-6
    C_shunt = C_C_shunt * 1e-9
    C_pi_t = C_pi_t * 1e-9
    L_ant = L_ant_muH * 1e-6
    C_ant = C_ant_nF * 1e-9

    # Frequenzvektor generieren
    f_hz = np.linspace(f_min * 1e6, f_max * 1e6, num_f_points)
    
    # Roh-Antennenimpedanz (ohne Netzwerk)
    Z_ant = compute_antenna_impedance(f_hz, R_ant, L_ant, C_ant, X0)
    gamma_ant = compute_gamma(Z_ant, Z0)
    
    # Transformiere mit gewähltem Netzwerk
    if network_type == 'None':
        Z_total = Z_ant
    elif network_type == 'L Network (Series L)':
        Z_total = series_L_transform(Z_ant, f_hz, L_series)
    elif network_type == 'C Network (Shunt C)':
        Z_total = shunt_C_transform(Z_ant, f_hz, C_shunt)
    elif network_type == 'Series L + Shunt C':
        Z_total = series_L_shunt_C_transform(Z_ant, f_hz, L_series, C_shunt)
    elif network_type == 'Shunt L + Series C':
        Z_total = shunt_L_series_C_transform(Z_ant, f_hz, L_shunt, C_pi_t)
    elif network_type == 'Pi Network (L1, L2, C)':
        Z_total = pi_network_transform(Z_ant, f_hz, L1, L2, C_pi_t)
    elif network_type == 'T Network (L1, L2, C)':
        Z_total = t_network_transform(Z_ant, f_hz, L1, L2, C_pi_t)
    else:
        Z_total = Z_ant
    
    # Reflexionskoeffizient nach Netzwerk
    gamma_total = compute_gamma(Z_total, Z0)
    
    # Plot initialisieren
    fig, (ax_smith, ax_rl, ax_z) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Smith-Chart
    plot_smith_chart(ax_smith, Z0)
    ax_smith.plot(gamma_ant.real, gamma_ant.imag, 'b--', lw=1.5, label='Antenne (ohne Match)')
    ax_smith.plot(gamma_total.real, gamma_total.imag, 'r', lw=2, label=f'Netzwerk: {network_type}')
    ax_smith.legend(fontsize=10)
    
    # Return-Loss-Spektrum
    plot_return_loss(ax_rl, f_hz, gamma_total, gamma_ant, Z0)
    
    # Normierte Impedanz
    plot_normalized_impedance(ax_z, f_hz, Z_total, Z_ant, Z0)
    
    plt.tight_layout(), plt.show()

# =================================================================
# 4. Interaktive Widgets (nur in Jupyter Notebook)
# =================================================================
interact(
    update_plots,
    
    # Frequenzbereich (MHz)
    f_min=FloatSlider(min=1, max=100, step=0.5, value=5, description='Min Frequenz:'),
    f_max=FloatSlider(min=1, max=100, step=0.5, value=20, description='Max Frequenz:'),
    num_f_points=IntSlider(min=100, max=2000, step=100, value=1000, description='Frequenzpunkte:'),
    
    # Referenzimpedanz (Ω)
    Z0=FloatSlider(min=25, max=100, step=5, value=50, description='Z0:'),
    
    # Netzwerk-Typ
    network_type=Dropdown(
        options=['None', 'L Network (Series L)', 'C Network (Shunt C)',
                 'Series L + Shunt C', 'Shunt L + Series C',
                 'Pi Network (L1, L2, C)', 'T Network (L1, L2, C)'],
        value='None', description='Netzwerk-Typ:'
    ),
    
    # L-Parameter (µH)
    L_L_series=FloatSlider(min=0, max=100, step=0.1, value=1, description='Series L:'),
    L_C_shunt=FloatSlider(min=0, max=100, step=0.1, value=1, description='Shunt L:'),
    L1_pi_t=FloatSlider(min=0, max=100, step=0.1, value=1, description='Pi/T L1:'),
    L2_pi_t=FloatSlider(min=0, max=100, step=0.1, value=1, description='Pi/T L2:'),
    
    # C-Parameter (nF)
    C_C_shunt=FloatSlider(min=0, max=100, step=0.1, value=1, description='Shunt C:'),
    C_pi_t=FloatSlider(min=0, max=100, step=0.1, value=1, description='Pi/T C:'),
    
    # Antennenparameter
    R_ant=FloatSlider(min=10, max=100, step=5, value=50, description='R_ant (Ω):'),
    L_ant_muH=FloatSlider(min=0, max=100, step=0.1, value=1, description='L_ant (µH):'),
    C_ant_nF=FloatSlider(min=0, max=100, step=0.1, value=1, description='C_ant (nF):'),
    X0=FloatSlider(min=-100, max=100, step=1, value=0, description='X0 (Ω):')
);

# Hinweis: Diese interaktive Visualisierung ist für Jupyter Notebooks gedacht.