"""
3D Antenna Directivity Pattern Visualization Toolkit

Simulates and visualizes 3D directivity patterns (Gain vs. Θ/Φ) for various antenna types.
Supported types: Isotropic, Dipole, Linear Array, Yagi-Uda. Interactive parameter controls.

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
# Antenna Gain Calculation Functions
# ---------------------------

def isotropic_gain(theta_grid, phi_grid):
    """
    Calculate gain for an isotropic antenna (uniform in all directions).
    
    Args:
        theta_grid (np.ndarray): 2D grid of vertical angles (rad, shape (Nθ, Nφ))
        phi_grid (np.ndarray): 2D grid of horizontal angles (rad, shape (Nθ, Nφ))
    
    Returns:
        np.ndarray: Linear gain (shape (Nθ, Nφ), values = 1)
    """
    return np.ones_like(theta_grid)  # Constant gain (no directionality)

def dipole_gain(theta_grid, phi_grid, frequency, dipole_length):
    """
    Calculate gain for a half-wave dipole antenna.
    Avoids division-by-zero at θ=0 (vertical angle 0 rad = straight up).
    
    Args:
        theta_grid (np.ndarray): 2D vertical angles grid (rad, (Nθ, Nφ))
        phi_grid (np.ndarray): 2D horizontal angles grid (rad, (Nθ, Nφ))
        frequency (float): Operating frequency (Hz)
        dipole_length (float): Total dipole length (m)
    
    Returns:
        np.ndarray: Linear gain (shape (Nθ, Nφ))
    """
    wavelength = 3e8 / frequency  # Free-space wavelength (m)
    norm_length = dipole_length / wavelength  # Normalized length (L/λ)
    
    arg = np.pi * norm_length * np.sin(theta_grid)  # Argument for trigonometric functions
    denominator = np.sin(arg)  # Could be zero at θ=0 (if norm_length=0.5, arg=π/2 * sinθ)
    
    # Avoid division by zero: if denominator ≈ 0 → set to 1e-9 (small non-zero)
    denominator = np.where(np.abs(denominator) < 1e-9, 1e-9, denominator)
    
    # Vertical gain component (cardioid pattern for λ/2 dipole)
    gain_vertical = np.sin(theta_grid) * (np.cos(arg) / denominator) ** 2
    
    # Horizontal gain component (omnidirectional, no φ-dependence)
    gain_horizontal = np.ones_like(phi_grid)
    
    return gain_vertical * gain_horizontal  # Total gain (θ-dependent, φ-independent)

def linear_array_gain(theta_grid, phi_grid, frequency, num_elements, element_spacing, phase_shift, dipole_length):
    """
    Calculate gain for a linear (1D) antenna array (Uniform Linear Array - ULA).
    Elements are arranged along the theta-axis (vertical direction).
    
    Args:
        theta_grid (np.ndarray): 2D vertical angles grid (rad, (Nθ, Nφ))
        phi_grid (np.ndarray): 2D horizontal angles grid (rad, (Nθ, Nφ))
        frequency (float): Operating frequency (Hz)
        num_elements (int): Number of array elements (N ≥ 1)
        element_spacing (float): Distance between adjacent elements (m)
        phase_shift (float): Phase delay per element (rad, applies to each element after the first)
        dipole_length (float): Length of each array element (m, treated as dipole)
    
    Returns:
        np.ndarray: Linear gain (shape (Nθ, Nφ))
    """
    wavelength = 3e8 / frequency  # Wavelength (m)
    k = 2 * np.pi / wavelength    # Wave number (rad/m)
    
    # Phase contribution per element (without element index) for each (θ, φ)
    phase_per_element = k * element_spacing * np.sin(theta_grid) + phase_shift
    # Expand to 3D (Nθ, Nφ, 1) to broadcast with element indices (0..N-1)
    phase_per_element_3d = phase_per_element[..., np.newaxis]  # Shape (Nθ, Nφ, 1)
    
    # Element indices (0 to num_elements-1) → shape (num_elements,)
    element_indices = np.arange(num_elements)
    
    # Phase for each element: phase_per_element_3d * element_indices → (Nθ, Nφ, N)
    phase = phase_per_element_3d * element_indices
    
    # Array Factor (AF): sum of complex exponentials over all elements
    array_factor = np.sum(np.exp(1j * phase), axis=-1)  # Sum over elements (axis=-1), shape (Nθ, Nφ)
    
    # Single element gain (dipole)
    gain_single = dipole_gain(theta_grid, phi_grid, frequency, dipole_length)
    
    # Total array gain: element gain * |AF|² (power combining, not amplitude)
    gain_linear = gain_single * np.abs(array_factor) ** 2
    
    return gain_linear

def yagi_gain(theta_grid, phi_grid, frequency, dipole_lengths, director_positions, element_spacing):
    """
    Simplified gain calculation for a Yagi-Uda antenna.
    Includes driven element, reflector, and directors (element positions defined via spacing).
    
    Args:
        theta_grid (np.ndarray): 2D vertical angles grid (rad, (Nθ, Nφ))
        phi_grid (np.ndarray): 2D horizontal angles grid (rad, (Nθ, Nφ))
        frequency (float): Operating frequency (Hz)
        dipole_lengths (list): Lengths of elements [driven, reflector, director1, ...] (m)
        director_positions (list): Positions of directors relative to driven element (m, positive = forward)
        element_spacing (float): Spacing between elements (m, used for reflector/director positioning)
    
    Returns:
        np.ndarray: Linear gain (shape (Nθ, Nφ), normalized to max gain=1)
    """
    wavelength = 3e8 / frequency  # Wavelength (m)
    gain = np.zeros_like(theta_grid, dtype=complex)  # Complex gain for summation
    
    # Driven element (first in list)
    if len(dipole_lengths) >= 1:
        driven_gain = dipole_gain(theta_grid, phi_grid, frequency, dipole_lengths[0])
        gain += driven_gain  # No phase shift (reference)
    
    # Reflector (second element, if present)
    if len(dipole_lengths) >= 2:
        reflector_length = dipole_lengths[1]
        # Reflector position: typically behind driven element (negative direction)
        # Using element_spacing to define distance (simplified)
        reflector_pos = -element_spacing  # Position relative to driven element (m)
        # Phase shift for reflector (redirects backward radiation forward)
        phase_reflector = -2 * np.pi * (np.abs(reflector_pos) / wavelength) * np.cos(theta_grid)
        reflector_gain = dipole_gain(theta_grid, phi_grid, frequency, reflector_length) * np.exp(1j * phase_reflector)
        gain += reflector_gain
    
    # Directors (remaining elements, if present)
    for i in range(2, len(dipole_lengths)):
        if i-2 >= len(director_positions):
            # Fallback if not enough positions defined (use element_spacing * i)
            dir_pos = element_spacing * i
        else:
            dir_pos = director_positions[i-2]
        dir_length = dipole_lengths[i]
        # Phase shift for director (focuses forward energy)
        phase_dir = 2 * np.pi * (dir_pos / wavelength) * np.cos(theta_grid)
        dir_gain = dipole_gain(theta_grid, phi_grid, frequency, dir_length) * np.exp(1j * phase_dir)
        gain += dir_gain
    
    # Convert complex sum to linear gain (magnitude squared)
    gain_linear = np.abs(gain) ** 2
    
    # Normalize to max gain=1 (avoid division by zero if all gains are zero)
    if np.max(gain_linear) == 0:
        return np.ones_like(gain_linear)
    return gain_linear / np.max(gain_linear)

# ---------------------------
# 3D Pattern Plotting Function
# ---------------------------

def plot_3d_directivity(theta, phi, gain_db, antenna_type):
    """
    Generate 3D directivity plot (gain vs. Θ/Φ) and a 2D horizontal cross-section.
    Gain is color-mapped and shown as a normalized sphere.
    
    Args:
        theta (np.ndarray): 1D array of vertical angles (rad, length Nθ)
        phi (np.ndarray): 1D array of horizontal angles (rad, length Nφ)
        gain_db (np.ndarray): Gain in dB (shape (Nθ, Nφ))
        antenna_type (str): Antenna description for plot titles
    
    Returns:
        None: Displays plots
    """
    # Create 3D meshgrid from theta and phi (needed for plot_surface)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')  # Shape (Nθ, Nφ)
    
    # Normalize gain for radius (0 ≤ r ≤ 1)
    max_gain_db = np.max(gain_db)
    if max_gain_db == -np.inf or max_gain_db == 0:
        # Handle edge case (no gain or all gain zero)
        r = np.ones_like(gain_db)
    else:
        # Gain ratio (linear): 10^(gain_db / 20) → but normalized to max gain
        gain_ratio = 10 ** (gain_db / 20) / (10 ** (max_gain_db / 20))
        r = np.sqrt(gain_ratio)  # Radius proportional to sqrt(gain) for better visibility
    
    # Convert spherical to Cartesian coordinates
    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.sin(theta_grid) * np.sin(phi_grid)
    z = r * np.cos(theta_grid)
    
    # Create figure with 3D plot and 2D cross-section
    fig = plt.figure(figsize=(18, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)
    
    # 3D Surface Plot
    # Use viridis colormap (lower gain = purple, higher gain = yellow)
    surf = ax3d.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        facecolors=plt.cm.viridis(gain_db / max_gain_db),  # Color mapped to gain ratio
        linewidth=0,
        antialiased=False
    )
    surf.set_clim(vmin=-60, vmax=0)  # Color scale limits (dB)
    
    # Add color bar (gain in dB)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=-60, vmax=0)),
        ax=ax3d, shrink=0.8, pad=0.05, label='Gain (dB)'
    )
    cbar.set_ticks(np.linspace(-60, 0, 7))  # Explicit ticks for readability
    
    # 3D Plot Settings
    ax3d.set_xlabel('X (Normalized)')
    ax3d.set_ylabel('Y (Normalized)')
    ax3d.set_zlabel('Z (Normalized)')
    ax3d.set_title(f'3D Directivity Pattern ({antenna_type})')
    ax3d.grid(False)
    ax3d.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax3d.view_init(elev=30, azim=45)  # Default view angle (adjustable via Jupyter widgets)
    
    # 2D Horizontal Cross-Section (φ=0)
    idx_phi0 = np.argmin(np.abs(phi - 0))  # Find φ=0 index
    theta_deg_phi0 = np.degrees(theta)  # Convert theta to degrees
    gain_db_phi0 = gain_db[:, idx_phi0]  # Extract gain at φ=0 (shape (Nθ,))
    
    ax2d.plot(theta_deg_phi0, gain_db_phi0, color='black', linewidth=2)
    ax2d.set_xlabel('Vertical Angle Θ (deg)')
    ax2d.set_ylabel('Gain (dB)')
    ax2d.set_title(f'Horizontal Cross-Section (φ=0) for {antenna_type}')
    ax2d.grid(True, linestyle='--')
    ax2d.set_xlim(-90, 90)  # Theta from -90 (down) to 90 (up) deg
    ax2d.set_ylim(-60, 0)  # Match colorbar limits
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Interactive Simulation Function
# ---------------------------

def interactive_directivity_simulation(
    frequency=900e6,
    antenna_type='Isotropic',
    dipole_length=0.5,
    num_elements=3,
    element_spacing=0.1,
    phase_shift=0.0,
    dipole_lengths_key="Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)",
    director_positions_key="Breit (0.1m, 0.2m)"
):
    """
    Main interactive function: Simulate and visualize 3D directivity patterns.
    Uses user-defined parameters to compute gain and trigger plotting.
    
    Args:
        frequency (float): Operating frequency (Hz)
        antenna_type (str): Selected antenna type (options: 'Isotropic', 'Dipole', 'Linear Array', 'Yagi')
        dipole_length (float): Length of a single dipole element (m) (used for Dipole and Arrays)
        num_elements (int): Number of elements in linear array (N) (used for Linear Array)
        element_spacing (float): Spacing between array elements (m) (used for Linear Array and Yagi)
        phase_shift (float): Phase delay per element (rad) (used for Linear Array)
        dipole_lengths_key (str): Key to Yagi element lengths dictionary (defines driven/reflector/directors)
        director_positions_key (str): Key to Yagi director positions dictionary (defines director locations)
    
    Returns:
        None: Displays interactive plots
    """
    # Angle ranges (cover full sphere)
    theta = np.linspace(0, np.pi, 100)  # Vertical angles (0=up, π=down), 100 points
    phi = np.linspace(0, 2 * np.pi, 100)  # Horizontal angles (0 to 2π), 100 points
    
    # Map string keys to actual lists (Yagi parameters)
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
    dipole_lengths = YAGI_DIPOLE_LENGTHS[dipole_lengths_key]
    director_positions = DIRECTOR_POSITIONS[director_positions_key]
    
    # Compute gain based on antenna type
    if antenna_type == 'Isotropic':
        gain_linear = isotropic_gain(theta, phi)
    elif antenna_type == 'Dipole':
        gain_linear = dipole_gain(theta, phi, frequency, dipole_length)
    elif antenna_type == 'Linear Array':
        # Ensure dipole_length is valid (used for array elements)
        if dipole_length <= 0:
            dipole_length = 0.5  # Fallback to standard length if invalid
        gain_linear = linear_array_gain(theta, phi, frequency, num_elements, element_spacing, phase_shift, dipole_length)
    elif antenna_type == 'Yagi':
        # Check if at least driven and reflector elements are defined
        if len(dipole_lengths) < 2:
            dipole_lengths = YAGI_DIPOLE_LENGTHS["Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)"]
        gain_linear = yagi_gain(theta, phi, frequency, dipole_lengths, director_positions, element_spacing)
    else:
        gain_linear = np.ones_like(theta)  # Fallback for unknown types
    
    # Convert linear gain to dB and clip to avoid -inf (invalid values)
    if np.max(gain_linear) == 0:
        gain_db = np.zeros_like(gain_linear) - 60  # All gains = -60 dB (minimum)
    else:
        gain_db = 20 * np.log10(gain_linear)
        gain_db = np.clip(gain_db, -60, 0)  # Clip to -60 dB (minimum visible)
    
    # Plot 3D directivity pattern
    plot_3d_directivity(theta, phi, gain_db, antenna_type)

# ---------------------------
# Interactive Widgets Setup
# ---------------------------

# Define widgets with clear descriptions and ranges
interact(
    interactive_directivity_simulation,
    frequency=FloatSlider(
        min=1e6, max=6e9, step=1e6, value=900e6,
        description='Frequency (Hz):'
    ),
    antenna_type=Dropdown(
        options=['Isotropic', 'Dipole', 'Linear Array', 'Yagi'],
        value='Isotropic',
        description='Antenna Type:'
    ),
    dipole_length=FloatSlider(
        min=0.1, max=2.0, step=0.05, value=0.5,
        description='Single Dipole Length (m):'
    ),
    num_elements=IntSlider(
        min=1, max=10, step=1, value=3,
        description='Linear Array Elements (N):'
    ),
    element_spacing=FloatSlider(
        min=0.01, max=0.5, step=0.01, value=0.1,
        description='Element Spacing (m):'
    ),
    phase_shift=FloatSlider(
        min=-np.pi, max=np.pi, step=0.1, value=0.0,
        description='Element Phase Shift (rad):'
    ),
    dipole_lengths_key=Dropdown(
        options=list(YAGI_DIPOLE_LENGTHS.keys()),
        value="Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)",
        description='Yagi Element Lengths:'
    ),
    director_positions_key=Dropdown(
        options=list(DIRECTOR_POSITIONS.keys()),
        value="Breit (0.1m, 0.2m)",
        description='Yagi Director Positions (m):'
    )
);