"""
HF Impedance Matching Simulation & Visualization Toolkit

This script simulates and visualizes impedance behavior of an RF antenna with various matching networks.
It combines theoretical calculations (impedance, reflection coefficient, return loss) with interactive plots
(Smith chart, return loss spectrum, normalized impedance components) to explore how network configurations
affect impedance matching across a frequency range.

Key Principles:
- Focus on "Why" over "What": Comments explain design intent and engineering context.
- Avoid Redundancy: No comments for self-explanatory code (e.g., "omega = 2πf" is obvious from variable names).
- Consistent Style: PEP 257-compliant docstrings for functions; clear axis/plot labels.
- Clarify Complex Logic: Workarounds (e.g., R_ant fallback) and non-intuitive calculations are documented.

Requirements:
- Python 3.10+
- numpy, matplotlib, ipywidgets (install via pip)

Author: Richard Zuikov
Date: 24.11.2025
Copyright: Just Enjoy the Toolkit
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown

def compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0):
    """
    Calculate antenna input impedance over frequency, including resistive, inductive, and capacitive components.
    Models the antenna as a lumped element: Z_ant = R_ant + j*(X0 + ω*L_ant - 1/(ω*C_ant)).
    The reactive term combines parasitic reactance (X0) with series inductance (L_ant) and shunt capacitance (C_ant).

    Args:
        f (np.ndarray): Frequency array (Hz)
        R_ant (float): Antenna resistive impedance (Ω, must be >0; fallback to 1e-3 Ω if ≤0)
        L_ant (float): Series inductance of the antenna (H)
        C_ant (float): Shunt capacitance of the antenna (F, treated as 0 if ≤0 to avoid division by zero)
        X0 (float): Additional parasitic reactance (Ω, e.g., for dielectric losses)

    Returns:
        np.ndarray: Complex antenna impedance array (Ω)
    """
    if R_ant <= 0:
        R_ant = 1e-3  # Fallback for numerical stability
    
    omega = 2 * np.pi * f
    X_L = omega * L_ant
    X_C = -1/(omega * C_ant) if C_ant > 0 else 0
    X_ant = X0 + X_L + X_C
    
    return R_ant + 1j * X_ant

def series_L_transform(Z_in, f, L_L):
    """
    Adjust input impedance with a series inductor for impedance matching.
    Adds inductive impedance (jωL_L) to Z_in to compensate for capacitive reactance.

    Args:
        Z_in (np.ndarray): Input impedance (Ω, complex array)
        f (np.ndarray): Frequency array (Hz)
        L_L (float): Series inductor value (H)

    Returns:
        np.ndarray: Transformed impedance (Ω, complex array)
    """
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L
    return Z_in + Z_L

def shunt_C_transform(Z_in, f, C_C):
    """
    Adjust input impedance with a shunt (parallel) capacitor for impedance matching.
    Adds capacitive reactance to counteract inductive impedance via parallel capacitance.

    Args:
        Z_in (np.ndarray): Input impedance (Ω, complex array)
        f (np.ndarray): Frequency array (Hz)
        C_C (float): Shunt capacitor value (F, treated as 0 if ≤0 → no effect)

    Returns:
        np.ndarray: Transformed impedance (Ω, complex array)
    """
    if C_C <= 0:
        return Z_in
    
    omega = 2 * np.pi * f
    Z_C = 1/(1j * omega * C_C)
    return (Z_in * Z_C) / (Z_in + Z_C)

def series_L_shunt_C_transform(Z_in, f, L_L, C_C):
    """
    Combine series inductor (L_L) and shunt capacitor (C_C) for impedance transformation.
    First apply series L, then parallel C to fine-tune reactive balance.

    Args:
        Z_in (np.ndarray): Input impedance (Ω, complex array)
        f (np.ndarray): Frequency array (Hz)
        L_L (float): Series inductor value (H)
        C_C (float): Shunt capacitor value (F)

    Returns:
        np.ndarray: Transformed impedance (Ω, complex array)
    """
    Z_series_L = series_L_transform(Z_in, f, L_L)
    return shunt_C_transform(Z_series_L, f, C_C)

def shunt_L_series_C_transform(Z_in, f, L_L, C_C):
    """
    Transform input impedance with a shunt inductor (L_L) and series capacitor (C_C) network.
    Rarely used but included for completeness. Shunt L adds parallel inductance; series C adds series capacitance.

    Args:
        Z_in (np.ndarray): Input impedance (Ω, complex array)
        f (np.ndarray): Frequency array (Hz)
        L_L (float): Shunt inductor value (H, treated as 0 if ≤0 → no effect)
        C_C (float): Series capacitor value (F, treated as 0 if ≤0 → open circuit)

    Returns:
        np.ndarray: Transformed impedance (Ω, complex array)
    """
    if L_L <= 0:
        return Z_in
    
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L  # Impedance of shunt inductor
    Z_parallel_L = (Z_in * Z_L) / (Z_in + Z_L)  # Z_in || Z_L
    
    # Add series capacitor (only if C_C > 0; else: Z_parallel_L remains unchanged)
    if C_C > 0:
        Z_Series_C = Z_parallel_L + 1/(1j * omega * C_C)  # Series C impedance: 1/(jωC)
    else:
        Z_Series_C = Z_parallel_L
    
    return Z_Series_C  # Korrigierter Variablentausch: Leerzeichen entfernt

def pi_network_transform(Z_in, f, L1, L2, C):
    """
    Transform input impedance with a π-type matching network (L1 series, C shunt, L2 series).
    Common for broadband matching. Configuration: Z_in → L1 → (L2 || C) → output.

    Args:
        Z_in (np.ndarray): Input impedance (Ω, complex array)
        f (np.ndarray): Frequency array (Hz)
        L1 (float): First series inductor (H)
        L2 (float): Second series inductor (H)
        C (float): Shunt capacitor (F, treated as 0 if ≤0 → open circuit)

    Returns:
        np.ndarray: Transformed impedance (Ω, complex array)
    """
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1
    Z_L2 = 1j * omega * L2
    
    if C <= 0:
        Z_C = np.inf  # Open circuit (no shunt C effect)
    else:
        Z_C = 1/(1j * omega * C)
    
    # Compute Z_parallel: L2 || C (parallel combination)
    Z_parallel = (Z_L2 * Z_C) / (Z_L2 + Z_C) if Z_C != np.inf else Z_L2
    
    # Total π-network impedance: Z_in + L1 + Z_parallel
    return Z_in + Z_L1 + Z_parallel

def t_network_transform(Z_in, f, L1, L2, C):
    """
    Transform input impedance with a T-type matching network (L1 shunt, C series, L2 shunt).
    Alternative to π-networks for impedance transformation.

    Args:
        Z_in (np.ndarray): Input impedance (Ω, complex array)
        f (np.ndarray): Frequency array (Hz)
        L1 (float): First shunt inductor (H, treated as 0 if ≤0 → no effect)
        L2 (float): Second shunt inductor (H, treated as 0 if ≤0 → no effect)
        C (float): Series capacitor (F, treated as 0 if ≤0 → open circuit)

    Returns:
        np.ndarray: Transformed impedance (Ω, complex array)
    """
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1 if L1 > 0 else np.inf  # Shunt L1 impedance (or open if invalid)
    
    if C <= 0:
        Z_C = np.inf  # Open circuit (no series C effect)
    else:
        Z_C = 1/(1j * omega * C)  # Series C impedance
    
    # Z1: Z_in || L1 (parallel with shunt L1)
    if Z_L1 != np.inf:
        Z1 = (Z_in * Z_L1) / (Z_in + Z_L1)
    else:
        Z1 = Z_in
    
    # Z2: Z1 + C (series with capacitor)
    if Z_C != np.inf:
        Z2 = Z1 + Z_C
    else:
        Z2 = Z1
    
    # Total Z: Z2 || L2 (parallel with shunt L2)
    if L2 > 0:
        Z_L2 = 1j * omega * L2
        Z_total = (Z2 * Z_L2) / (Z2 + Z_L2)
    else:
        Z_total = Z2
    
    return Z_total

def compute_total_impedance(f, antenna_impedance_func, network_type, L_L, C_C, L1, L2, C, Z0):
    """
    Compute total impedance after applying the selected matching network.
    Dispatches impedance transformation based on network_type.

    Args:
        f (np.ndarray): Frequency array (Hz)
        antenna_impedance_func (callable): Function to compute raw antenna impedance (signature: Z_ant = func(f))
        network_type (str): Matching network type (options: 'None', 'L Network (Series L)', ...)
        L_L (float): Series inductor (H) - used for L-type networks
        C_C (float): Shunt capacitor (F) - used for C-type networks
        L1 (float): First inductor (H) - used for Pi/T-type networks
        L2 (float): Second inductor (H) - used for Pi/T-type networks
        C (float): Capacitor (F) - used for Pi/T-type networks
        Z0 (float): System reference impedance (Ω)

    Returns:
        np.ndarray: Total impedance (Ω, complex array)
    """
    Z_ant = antenna_impedance_func(f)  # Raw antenna impedance
    
    if network_type == 'None':
        Z_total = Z_ant
    elif network_type == 'L Network (Series L)':
        Z_total = series_L_transform(Z_ant, f, L_L)
    elif network_type == 'C Network (Shunt C)':
        Z_total = shunt_C_transform(Z_ant, f, C_C)
    elif network_type == 'Series L + Shunt C':
        Z_total = series_L_shunt_C_transform(Z_ant, f, L_L, C_C)
    elif network_type == 'Shunt L + Series C':
        Z_total = shunt_L_series_C_transform(Z_ant, f, L_L, C_C)
    elif network_type == 'Pi Network (L1, L2, C)':
        Z_total = pi_network_transform(Z_ant, f, L1, L2, C)
    elif network_type == 'T Network (L1, L2, C)':
        Z_total = t_network_transform(Z_ant, f, L1, L2, C)
    else:
        Z_total = Z_ant  # Fallback for unknown network types
    
    return Z_total

def compute_gamma(Z_total, Z0):
    """
    Calculate reflection coefficient Γ for impedance analysis.
    Γ = (Z_total/Z0 - 1)/(Z_total/Z0 + 1). Γ=0 indicates perfect match (Z_total=Z0).

    Args:
        Z_total (np.ndarray): Total impedance (Ω, complex array)
        Z0 (float): Reference impedance (Ω)

    Returns:
        np.ndarray: Reflection coefficient Γ (complex array)
    """
    Z_norm = Z_total / Z0
    return (Z_norm - 1) / (Z_norm + 1)

def compute_return_loss(gamma):
    """
    Convert reflection coefficient Γ to return loss RL (dB).
    RL(dB) = 20*log10(|Γ|). Better matching → higher RL (less negative).

    Args:
        gamma (np.ndarray): Reflection coefficient (complex array)

    Returns:
        np.ndarray: Return loss array (dB)
    """
    return 20 * np.log10(np.abs(gamma))

def plot_smith_chart(ax, Z0):
    """
    Plot Smith chart grid (constant R/X lines) on a matplotlib axis.
    Visualizes normalized impedance (Z/Z0) as Γ trajectories. Grid lines show constant R_norm/X_norm.

    Args:
        ax (matplotlib.axes.Axes): Target axis
        Z0 (float): Reference impedance (Ω)

    Returns:
        None: Modifies axis in-place
    """
    theta = np.linspace(0, 2*np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', linewidth=1)  # Unit circle
    
    # Constant R_norm circles (R=0 to R=10Ω, key ranges for readability)
    R_norm_vals = np.concatenate([np.arange(0, 2, 0.5), np.arange(2.5, 10, 0.5)])
    for R_norm in R_norm_vals:
        X_norm_vals = np.linspace(-10, 10, 1000)
        Z_norm = R_norm + 1j * X_norm_vals
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, color='gray', linestyle='-', linewidth=0.5)
    
    # Constant X_norm lines (X=-10 to X=+10Ω, fine grid)
    X_norm_vals = np.arange(-10, 10, 0.5)
    for X_norm in X_norm_vals:
        R_norm_vals = np.linspace(0, 100, 1000)
        Z_norm = R_norm_vals + 1j * X_norm
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Γ Real Part')
    ax.set_ylabel('Γ Imaginary Part')
    ax.set_title('Smith Chart (Impedance Matching)')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(False)
    ax.set_aspect('equal')

def plot_smith_and_matching(f_min=5, f_max=20, num_f=1000, Z0=50,
                            network_type='None', L_L=1e-6, C_C=1e-9,
                            L1=1e-6, L2=1e-6, C=1e-9,
                            R_ant=50, L_ant=1e-6, C_ant=1e-9, X0=0):
    """
    Main function: Simulate and visualize impedance matching across frequency.
    Generates Smith chart, return loss spectrum, and normalized impedance components.

    Args:
        f_min (float): Start frequency (MHz)
        f_max (float): End frequency (MHz)
        num_f (int): Number of frequency points (slider returns float → converted to int)
        Z0 (float): Reference impedance (Ω, e.g., 50Ω for RF systems)
        network_type (str): Matching network type (from Dropdown options)
        L_L (float): Series L (H) - used for L-type networks
        C_C (float): Shunt C (F) - used for C-type networks
        L1 (float): Pi/T L1 (H)
        L2 (float): Pi/T L2 (H)
        C (float): Pi/T C (F)
        R_ant (float): Antenna resistance (Ω)
        L_ant (float): Antenna series L (H)
        C_ant (float): Antenna shunt C (F)
        X0 (float): Antenna parasitic reactance (Ω)

    Returns:
        None: Displays plots
    """
    num_f = int(num_f)
    num_f = max(num_f, 100)  # Ensure minimum 100 points for smooth plots
    
    f_min_hz = f_min * 1e6
    f_max_hz = f_max * 1e6
    f = np.linspace(f_min_hz, f_max_hz, num_f)
    
    # Compute raw antenna impedance (without matching)
    Z_ant = compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0)
    gamma_ant = compute_gamma(Z_ant, Z0)
    
    # Compute total impedance (with matching network)
    Z_total = compute_total_impedance(
        f,
        lambda f: compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0),
        network_type,
        L_L, C_C, L1, L2, C,
        Z0
    )
    gamma_total = compute_gamma(Z_total, Z0)
    
    # Return loss calculations
    return_loss_total = compute_return_loss(gamma_total)
    return_loss_ant = compute_return_loss(gamma_ant)
    
    # Create 3-panel figure
    fig, (ax_smith, ax_rl, ax_z) = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot Smith chart
    plot_smith_chart(ax_smith, Z0)
    ax_smith.plot(gamma_ant.real, gamma_ant.imag, color='blue', linewidth=2, label='Antenna (No Match)')
    ax_smith.plot(gamma_total.real, gamma_total.imag, color='red', linewidth=2, label=f'Total (Network: {network_type})')
    
    # Annotate antenna resonance (if L_ant and C_ant are valid)
    if L_ant > 0 and C_ant > 0:
        f0_hz = 1/(2*np.pi*np.sqrt(L_ant*C_ant))
        idx = np.argmin(np.abs(f - f0_hz))
        gamma_f0_ant = gamma_ant[idx]
        gamma_f0_total = gamma_total[idx]
        
        ax_smith.scatter(gamma_f0_ant.real, gamma_f0_ant.imag, color='blue', s=50, zorder=3)
        ax_smith.scatter(gamma_f0_total.real, gamma_f0_total.imag, color='red', s=50, zorder=3)
        
        ax_smith.text(gamma_f0_ant.real, gamma_f0_ant.imag, 
                     f'Antenna Resonance\nf0={f0_hz/1e6:.2f} MHz', 
                     color='blue', zorder=4, ha='right')
        ax_smith.text(gamma_f0_total.real, gamma_f0_total.imag, 
                     f'Network Resonance\nf0={f0_hz/1e6:.2f} MHz', 
                     color='red', zorder=4, ha='left')
    
    ax_smith.legend()
    
    # Plot return loss
    ax_rl.plot(f/1e6, return_loss_total, color='red', label='With Network')
    ax_rl.plot(f/1e6, return_loss_ant, color='blue', linestyle='--', label='Without Network')
    ax_rl.set_xlabel('Frequency (MHz)')
    ax_rl.set_ylabel('Return Loss (dB)')
    ax_rl.set_ylim(-60, 0)
    ax_rl.grid(True, linestyle='--')
    
    # Annotate worst/best RL points
    # Best RL (minimal loss) with network
    best_rl_total = np.max(return_loss_total)  # RL is max (least negative) at best match
    idx_best_total = np.argmax(return_loss_total)
    f_best_total = f[idx_best_total]/1e6
    ax_rl.scatter(f_best_total, best_rl_total, color='green', s=50, zorder=3)
    ax_rl.text(f_best_total, best_rl_total, 
              f'Best RL: {best_rl_total:.2f} dB\nat f: {f_best_total:.2f} MHz', 
              color='green', zorder=4, va='bottom')
    
    # Best RL (original antenna)
    best_rl_ant = np.max(return_loss_ant)
    idx_best_ant = np.argmax(return_loss_ant)
    f_best_ant = f[idx_best_ant]/1e6
    ax_rl.scatter(f_best_ant, best_rl_ant, color='orange', s=50, zorder=3)
    ax_rl.text(f_best_ant, best_rl_ant, 
              f'Antenna Best RL: {best_rl_ant:.2f} dB\nat f: {f_best_ant:.2f} MHz', 
              color='orange', zorder=4, va='bottom')
    
    ax_rl.legend()
    
    # Plot normalized impedance components (R/Z0, X/Z0)
    Z_total_norm = Z_total / Z0
    Z_ant_norm = Z_ant / Z0
    
    ax_z.plot(f/1e6, Z_ant_norm.real, color='blue', linestyle='--', label='Antenna R/Z0')
    ax_z.plot(f/1e6, Z_total_norm.real, color='green', label='Total R/Z0')
    ax_z.plot(f/1e6, Z_ant_norm.imag, color='red', linestyle='--', label='Antenna X/Z0')
    ax_z.plot(f/1e6, Z_total_norm.imag, color='purple', label='Total X/Z0')
    
    ax_z.set_xlabel('Frequency (MHz)')
    ax_z.set_ylabel('Normalized Impedance')
    ax_z.grid(True, linestyle='--')
    ax_z.legend()
    
    plt.tight_layout()
    plt.show()

# Interactive controls (Jupyter)
interact(
    plot_smith_and_matching,
    f_min=FloatSlider(min=1, max=100, step=0.5, value=5, description='Min Frequency (MHz)'),
    f_max=FloatSlider(min=1, max=100, step=0.5, value=20, description='Max Frequency (MHz)'),
    num_f=FloatSlider(min=100, max=2000, step=100, value=1000, description='Frequency Points'),
    Z0=FloatSlider(min=25, max=100, step=5, value=50, description='Reference Z0 (Ω)'),
    network_type=Dropdown(
        options=['None', 'L Network (Series L)', 'C Network (Shunt C)',
                 'Series L + Shunt C', 'Shunt L + Series C',
                 'Pi Network (L1, L2, C)', 'T Network (L1, L2, C)'],
        value='None',
        description='Network Type'
    ),
    L_L=FloatSlider(min=0, max=1e-5, step=1e-7, value=1e-6, description='Series L (H)'),
    C_C=FloatSlider(min=0, max=1e-8, step=1e-10, value=1e-9, description='Shunt C (F)'),
    L1=FloatSlider(min=0, max=1e-5, step=1e-7, value=1e-6, description='Pi/T L1 (H)'),
    L2=FloatSlider(min=0, max=1e-5, step=1e-7, value=1e-6, description='Pi/T L2 (H)'),
    C=FloatSlider(min=0, max=1e-8, step=1e-10, value=1e-9, description='Pi/T C (F)'),
    R_ant=FloatSlider(min=10, max=100, step=5, value=50, description='Antenna R (Ω)'),
    L_ant=FloatSlider(min=0, max=1e-5, step=1e-7, value=1e-6, description='Antenna L (H)'),
    C_ant=FloatSlider(min=0, max=1e-8, step=1e-10, value=1e-9, description='Antenna C (F)'),
    X0=FloatSlider(min=-100, max=100, step=1, value=0, description='Parasitic X0 (Ω)')
);
#  The Plotting graphs look good yeah!!