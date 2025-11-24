import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown

def compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0):
    if R_ant <= 0:
        R_ant = 1e-3
    omega = 2 * np.pi * f
    X_L = omega * L_ant
    X_C = -1/(omega * C_ant) if C_ant > 0 else 0
    X_ant = X0 + X_L + X_C
    return R_ant + 1j * X_ant

def series_L_transform(Z_in, f, L_L):
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L
    return Z_in + Z_L

def shunt_C_transform(Z_in, f, C_C):
    omega = 2 * np.pi * f
    Z_C = 1/(1j * omega * C_C) if C_C > 0 else 0
    return (Z_in * Z_C)/(Z_in + Z_C)

def series_L_shunt_C_transform(Z_in, f, L_L, C_C):
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L
    Z_C = 1/(1j * omega * C_C) if C_C > 0 else 0
    Z_in_series_L = Z_in + Z_L
    return (Z_in_series_L * Z_C)/(Z_in_series_L + Z_C)

def shunt_L_series_C_transform(Z_in, f, L_L, C_C):
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L
    Z_C = 1/(1j * omega * C_C) if C_C > 0 else 0
    Z_in_parallel_L = (Z_in * Z_L)/(Z_in + Z_L)
    return Z_in_parallel_L + Z_C

def pi_network_transform(Z_in, f, L1, L2, C):
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1
    Z_L2 = 1j * omega * L2
    Z_C = 1/(1j * omega * C) if C > 0 else 0
    Z_parallel = (Z_L2 * Z_C)/(Z_L2 + Z_C)
    return Z_in + Z_L1 + Z_parallel

def t_network_transform(Z_in, f, L1, L2, C):
    omega = 2 * np.pi * f
    Z_L1 = 1j * omega * L1
    Z_L2 = 1j * omega * L2
    Z_C = 1/(1j * omega * C) if C > 0 else 0
    Z1 = (Z_in * Z_L1)/(Z_in + Z_L1)
    Z2 = Z1 + Z_C
    return (Z2 * Z_L2)/(Z2 + Z_L2)

def compute_total_impedance(f, Z_ant_func, network_type, L_L, C_C, L1, L2, C, Z0):
    Z_ant = Z_ant_func(f)
    if network_type == 'None':
        Z_total = Z_ant
    elif network_type == 'L-Netz (Serie L)':
        Z_total = series_L_transform(Z_ant, f, L_L)
    elif network_type == 'C-Netz (Parallels C)':
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
        Z_total = Z_ant
    return Z_total

def compute_gamma(Z_total, Z0):
    Z_norm = Z_total / Z0
    return (Z_norm - 1)/(Z_norm + 1)

def compute_return_loss(gamma):
    return 20 * np.log10(np.abs(gamma))

def plot_smith_chart(ax, Z0):
    theta = np.linspace(0, 2*np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', linewidth=1)
    R_norm_vals = np.concatenate([np.arange(0, 2, 0.5), np.arange(2.5, 10, 0.5)])
    for R_norm in R_norm_vals:
        X_norm_vals = np.linspace(-10, 10, 1000)
        Z_norm = R_norm + 1j * X_norm_vals
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, color='gray', linestyle='-', linewidth=0.5)
    X_norm_vals = np.arange(-10, 10, 0.5)
    for X_norm in X_norm_vals:
        R_norm_vals = np.linspace(0, 100, 1000)
        Z_norm = R_norm_vals + 1j * X_norm
        gamma = (Z_norm - 1)/(Z_norm + 1)
        ax.plot(gamma.real, gamma.imag, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Realteil (Γ)')
    ax.set_ylabel('Imaginärteil (Γ)')
    ax.set_title('Smith-Chart')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(False)
    ax.set_aspect('equal', 'box')

def plot_smith_and_matching(f_min=5, f_max=20, num_f=1000, Z0=50,
                            network_type='None', L_L=1e-6, C_C=1e-9,
                            L1=1e-6, L2=1e-6, C=1e-9,
                            R_ant=50, L_ant=1e-6, C_ant=1e-9, X0=0):
    f_min_hz = f_min * 1e6
    f_max_hz = f_max * 1e6
    f = np.linspace(f_min_hz, f_max_hz, num_f)
    Z_ant = compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0)
    gamma_ant = compute_gamma(Z_ant, Z0)
    Z_total = compute_total_impedance(f, lambda f: compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0), network_type, L_L, C_C, L1, L2, C, Z0)
    gamma_total = compute_gamma(Z_total, Z0)
    return_loss_total = compute_return_loss(gamma_total)
    return_loss_ant = compute_return_loss(gamma_ant)
    fig, (ax_smith, ax_rl, ax_z) = plt.subplots(1, 3, figsize=(24, 8))
    plot_smith_chart(ax_smith, Z0)
    ax_smith.plot(gamma_ant.real, gamma_ant.imag, color='blue', linewidth=2)
    ax_smith.plot(gamma_total.real, gamma_total.imag, color='red', linewidth=2)
    f0_ant = 1/(2 * np.pi * np.sqrt(L_ant * C_ant)) if L_ant * C_ant > 0 else 0
    idx_f0 = np.argmin(np.abs(f - f0_ant)) if f0_ant !=0 else 0
    gamma_f0_total = gamma_total[idx_f0] if f0_ant !=0 else [0,0]
    gamma_f0_ant = gamma_ant[idx_f0] if f0_ant !=0 else [0,0]
    ax_smith.scatter(gamma_f0_total.real, gamma_f0_total.imag, color='green', s=50)
    ax_smith.scatter(gamma_f0_ant.real, gamma_f0_ant.imag, color='blue', s=50)
    ax_smith.text(gamma_f0_total.real, gamma_f0_total.imag, f'f0={f0_ant/1e6:.2f}MHz', color='green')
    ax_smith.text(gamma_f0_ant.real, gamma_f0_ant.imag, f'f0={f0_ant/1e6:.2f}MHz', color='blue')
    ax_rl.plot(f/1e6, return_loss_total, color='red', linewidth=2)
    ax_rl.plot(f/1e6, return_loss_ant, color='blue', linestyle='--', linewidth=1)
    ax_rl.set_xlabel('Frequenz (MHz)')
    ax_rl.set_ylabel('Return Loss (dB)')
    ax_rl.set_ylim(-60, 0)
    ax_rl.grid(True, linestyle='--')
    min_rl_total = np.min(return_loss_total)
    f_min_rl_total = f[np.argmin(return_loss_total)] / 1e6
    avg_rl_total = np.mean(return_loss_total)
    min_rl_ant = np.min(return_loss_ant)
    f_min_rl_ant = f[np.argmin(return_loss_ant)] / 1e6
    avg_rl_ant = np.mean(return_loss_ant)
    ax_rl.text(0.05, 0.95, f'Min RL (Netzwerk): {min_rl_total:.2f}dB\nbei f: {f_min_rl_total:.2f}MHz\nDurchschnitt: {avg_rl_total:.2f}dB',
              transform=ax_rl.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    ax_rl.text(0.05, 0.8, f'Min RL (Antenne): {min_rl_ant:.2f}dB\nbei f: {f_min_rl_ant:.2f}MHz\nDurchschnitt: {avg_rl_ant:.2f}dB',
              transform=ax_rl.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    Z_total_norm = Z_total / Z0
    Z_ant_norm = Z_ant / Z0
    ax_z.plot(f/1e6, Z_total_norm.real, color='green', linewidth=2)
    ax_z.plot(f/1e6, Z_total_norm.imag, color='purple', linewidth=2)
    ax_z.plot(f/1e6, Z_ant_norm.real, color='blue', linestyle='--', linewidth=1)
    ax_z.plot(f/1e6, Z_ant_norm.imag, color='red', linestyle='--', linewidth=1)
    ax_z.set_xlabel('Frequenz (MHz)')
    ax_z.set_ylabel('Normierte Impedanz (Z/Z0)')
    ax_z.grid(True, linestyle='--')
    ax_z.legend(['Realteil (Netzwerk)', 'Imaginärteil (Netzwerk)', 'Realteil (Antenne)', 'Imaginärteil (Antenne)'])
    ax_smith.legend(['Antenne', network_type])
    plt.tight_layout()
    plt.show()

interact(plot_smith_and_matching,
         f_min=FloatSlider(min=1, max=100, step=1, value=5, description='f_min (MHz)'),
         f_max=FloatSlider(min=1, max=100, step=1, value=20, description='f_max (MHz)'),
         num_f=FloatSlider(min=100, max=2000, step=100, value=1000, description='Anz. Punkte'),
         Z0=FloatSlider(min=25, max=100, step=5, value=50, description='Z0 (Ω)'),
         network_type=Dropdown(options=['None', 'L-Netz (Serie L)', 'C-Netz (Parallels C)',
                                        'Series L + Shunt C', 'Shunt L + Series C',
                                        'Pi Network (L1, L2, C)', 'T Network (L1, L2, C)'],
                               value='None', description='Netzwerk Typ'),
         L_L=FloatSlider(min=1e-9, max=1e-5, step=1e-7, value=1e-6, description='L_L (H)'),
         C_C=FloatSlider(min=1e-12, max=1e-8, step=1e-10, value=1e-9, description='C_C (F)'),
         L1=FloatSlider(min=1e-9, max=1e-5, step=1e-7, value=1e-6, description='L1 (H)'),
         L2=FloatSlider(min=1e-9, max=1e-5, step=1e-7, value=1e-6, description='L2 (H)'),
         C=FloatSlider(min=1e-12, max=1e-8, step=1e-10, value=1e-9, description='C (F)'),
         R_ant=FloatSlider(min=10, max=100, step=5, value=50, description='R_ant (Ω)'),
         L_ant=FloatSlider(min=1e-9, max=1e-5, step=1e-7, value=1e-6, description='L_ant (H)'),
         C_ant=FloatSlider(min=1e-12, max=1e-8, step=1e-10, value=1e-9, description='C_ant (F)'),
         X0=FloatSlider(min=-100, max=100, step=1, value=0, description='X0 (Ω)'));