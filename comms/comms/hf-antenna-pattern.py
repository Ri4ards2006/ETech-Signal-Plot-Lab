"""
3D Antenna Directivity Pattern Visualization Toolkit

This script simulates and visualizes 3D directivity patterns (gain vs. angles Θ/Φ) for various antenna types.
It supports isotropic emitters, dipoles, linear arrays, and Yagi-Uda antennas with interactive parameter controls.

Key Principles:
- Use of clear, PEP 257-compliant docstrings.
- Avoid complex types (lists) in widgets; map via dictionaries.
- Interactive exploration of antenna behavior via sliders/dropdowns.

Requirements:
- Python 3.10+
- numpy, matplotlib, ipywidgets (install via pip)

Author: [Your Name]
Date: [Today's Date]
Copyright: [Your Copyright Notice]
"""
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown, IntSlider

# ---------------------------
# Antenna Pattern Calculators
# ---------------------------

def isotropic_gain(theta, phi):
    return np.ones_like(theta)

def dipole_gain(theta, phi, frequency, dipole_length):
    wavelength = 3e8 / frequency
    norm_length = dipole_length / wavelength
    gain_vertical = np.sin(theta) * (np.cos(np.pi * norm_length * np.sin(theta)) / 
                                    np.sin(np.pi * norm_length * np.sin(theta)))**2
    gain_vertical = np.where(np.sin(theta) == 0, 1e-9, gain_vertical)  # Avoid division by zero at θ=0
    return gain_vertical * np.ones_like(phi)

def linear_array_gain(theta, phi, frequency, num_elements, element_spacing, phase_shift):
    wavelength = 3e8 / frequency
    k = 2 * np.pi / wavelength
    phase = k * element_spacing * np.sin(theta) + phase_shift
    af = np.sum(np.exp(1j * phase * np.arange(num_elements)), axis=0)  # Array Factor
    gain_single = np.sin(theta) if num_elements == 1 else dipole_gain(theta, phi, frequency, 0.5)  # Simplified single element gain
    return np.abs(gain_single * af)**2

def yagi_gain(theta, phi, frequency, dipole_lengths, director_positions):
    wavelength = 3e8 / frequency
    gain = np.zeros_like(theta, dtype=complex)
    
    # Driven element (first in list)
    driven_length = dipole_lengths[0]
    gain_driven = dipole_gain(theta, phi, frequency, driven_length)
    gain += gain_driven
    
    # Reflector (second in list, if exists)
    if len(dipole_lengths) >= 2:
        reflector_length = dipole_lengths[1]
        reflector_pos = -element_spacing  # Simplified: reflector behind driven element
        phase_reflector = -2 * np.pi * (np.abs(reflector_pos) / wavelength) * np.cos(theta)
        gain_reflector = dipole_gain(theta, phi, frequency, reflector_length) * np.exp(1j * phase_reflector)
        gain += gain_reflector
    
    # Directors (remaining in list)
    for i in range(2, len(dipole_lengths)):
        dir_length = dipole_lengths[i]
        dir_pos = director_positions[i-2]  # Positions start after driven/reflector
        phase_dir = 2 * np.pi * (dir_pos / wavelength) * np.cos(theta)
        gain_dir = dipole_gain(theta, phi, frequency, dir_length) * np.exp(1j * phase_dir)
        gain += gain_dir
    
    gain_linear = np.abs(gain)**2
    return gain_linear / np.max(gain_linear)  # Normalize to max gain=1

# ---------------------------
# 3D Pattern Plotting
# ---------------------------

def plot_3d_directivity(theta, phi, gain_db, antenna_type):
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian coordinates (normalized gain for radius)
    r = np.sqrt(gain_db / np.max(gain_db)) if np.max(gain_db) != 0 else np.ones_like(gain_db)
    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.sin(theta_grid) * np.sin(phi_grid)
    z = r * np.cos(theta_grid)
    
    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)
    
    # 3D Surface Plot
    surf = ax3d.plot_surface(x, y, z, 
                             rstride=1, cstride=1,
                             facecolors=plt.cm.viridis(gain_db / np.max(gain_db)),
                             linewidth=0, antialiased=False)
    surf.set_clim(vmin=np.min(gain_db), vmax=np.max(gain_db))
    
    # Colorbar
    plt.colorbar(surf, ax=ax3d, shrink=0.8, label='Gain (dB)')
    
    # 3D Settings
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title(f'3D Directivity: {antenna_type}')
    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-1, 1)
    
    # 2D Cross-Section (phi=0)
    idx_phi0 = np.argmin(np.abs(phi - 0))
    theta_deg = np.degrees(theta)
    gain_db_phi0 = gain_db[:, idx_phi0]
    ax2d.plot(theta_deg, gain_db_phi0, 'k-', linewidth=2)
    ax2d.set_xlabel('Vertical Angle Θ (deg)')
    ax2d.set_ylabel('Gain (dB)')
    ax2d.set_title(f'Horizontal Cross-Section (φ=0) for {antenna_type}')
    ax2d.grid(True, linestyle='--')
    ax2d.set_xlim(-90, 90)
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Interactive Widgets & Main Function
# ---------------------------

# Define options as dictionaries (display strings → actual lists)
YAGI_DIPOLE_LENGTHS = {
    "Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)": [0.5, 0.55, 0.48],
    "Mit 3 Directors": [0.5, 0.55, 0.48, 0.45],
    "Kurzwellenhaft (Higher Frequency)": [0.3, 0.32, 0.29]
}

DIRECTOR_POSITIONS = {
    "Standard (0.05m, 0.15m)": [0.05, 0.15],
    "Kompakt (0.03m, 0.08m)": [0.03, 0.08],
    "Breit (0.1m, 0.2m)": [0.1, 0.2]
}

def interactive_directivity_simulation(
    frequency=900e6,
    antenna_type='Isotropic',
    dipole_length=0.5,
    num_elements=3,
    element_spacing=0.1,
    phase_shift=0.0,
    dipole_lengths_key="Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)",
    director_positions_key="Standard (0.05m, 0.15m)"
):
    # Map strings to actual lists
    dipole_lengths = YAGI_DIPOLE_LENGTHS[dipole_lengths_key]
    director_positions = DIRECTOR_POSITIONS[director_positions_key]
    
    # Angle grids (0 ≤ θ ≤ π, 0 ≤ φ ≤ 2π)
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 100)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Calculate linear gain
    if antenna_type == 'Isotropic':
        gain_linear = isotropic_gain(theta_grid, phi_grid)
    elif antenna_type == 'Dipole':
        gain_linear = dipole_gain(theta_grid, phi_grid, frequency, dipole_length)
    elif antenna_type == 'Linear Array':
        gain_linear = linear_array_gain(theta_grid, phi_grid, frequency, num_elements, element_spacing, phase_shift)
    elif antenna_type == 'Yagi':
        # Ensure at least driven and reflector elements
        if len(dipole_lengths) < 2:
            dipole_lengths = YAGI_DIPOLE_LENGTHS["Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)"]
        gain_linear = yagi_gain(theta_grid, phi_grid, frequency, dipole_lengths, director_positions)
    else:
        gain_linear = np.ones_like(theta_grid)
    
    # Convert to dB and clip
    gain_db = 20 * np.log10(gain_linear)
    gain_db = np.clip(gain_db, -60, np.max(gain_db))
    
    # Plot
    plot_3d_directivity(theta, phi, gain_db, antenna_type)

# ---------------------------
# Interactive Widgets Setup
# ---------------------------

# Dropdowns with string keys
dipole_lengths_dropdown = Dropdown(
    options=list(YAGI_DIPOLE_LENGTHS.keys()),
    value="Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)",
    description='Yagi Element Lengths (m)'
)

director_positions_dropdown = Dropdown(
    options=list(DIRECTOR_POSITIONS.keys()),
    value="Standard (0.05m, 0.15m)",
    description='Director Positions (m)'
)

# Interact with all parameters
interact(
    interactive_directivity_simulation,
    frequency=FloatSlider(min=1e6, max=6e9, step=1e6, value=900e6, description='Frequency (Hz)'),
    antenna_type=Dropdown(options=['Isotropic', 'Dipole', 'Linear Array', 'Yagi'], value='Isotropic', description='Antenna Type'),
    dipole_length=FloatSlider(min=0.1, max=2.0, step=0.05, value=0.5, description='Dipole Length (m)'),
    num_elements=IntSlider(min=1, max=10, step=1, value=3, description='Array Elements (N)'),
    element_spacing=FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='Element Spacing (m)'),
    phase_shift=FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0.0, description='Phase Shift (rad)'),
    dipole_lengths_key=dipole_lengths_dropdown,
    director_positions_key=director_positions_dropdown
);