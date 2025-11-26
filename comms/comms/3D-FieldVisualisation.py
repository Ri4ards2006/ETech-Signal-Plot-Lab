"""
3D Antenna Directivity and Field Visualization Toolkit
Simulates 3D directivity patterns and electric/magnetic fields for various antenna types.
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

# Global definitions for Yagi parameters
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

# ---------------------------
# Antenna Gain Calculation Functions
# ---------------------------
def isotropic_gain(theta_grid, phi_grid):
    """Return constant gain (isotropic)"""
    return np.ones_like(theta_grid)

def dipole_gain(theta_grid, phi_grid, frequency, dipole_length):
    """Short dipole gain (l << λ): G(θ) = 1.5 * sin²θ"""
    wavelength = 3e8 / frequency
    if dipole_length > wavelength / 10:
        print(f"Warnung: Dipol ({dipole_length:.2f}m) ist nicht kurz (λ/10={wavelength/10:.2f}m).")
    return 1.5 * np.sin(theta_grid)**2 * np.ones_like(phi_grid)

def linear_array_gain(theta_grid, phi_grid, frequency, num_elements, element_spacing, phase_shift, dipole_length):
    """ULA gain with array factor"""
    wavelength = 3e8 / frequency
    k = 2 * np.pi / wavelength
    phase_per_element = k * element_spacing * np.sin(theta_grid) + phase_shift
    phase = phase_per_element[..., np.newaxis] * np.arange(num_elements)
    af = np.abs(np.sum(np.exp(1j * phase), axis=-1))**2
    return dipole_gain(theta_grid, phi_grid, frequency, dipole_length) * af

def yagi_gain(theta_grid, phi_grid, frequency, dipole_lengths, director_positions, element_spacing):
    """Simplified Yagi gain (sum of dipole contributions)"""
    wavelength = 3e8 / frequency
    gain = np.zeros_like(theta_grid, dtype=complex)
    if len(dipole_lengths) < 2:
        return isotropic_gain(theta_grid, phi_grid)
    
    # Driven element
    gain += dipole_gain(theta_grid, phi_grid, frequency, dipole_lengths[0])
    
    # Reflector (phase-shifted)
    reflector_pos = -element_spacing
    phase_reflector = -2*np.pi*(abs(reflector_pos)/wavelength)*np.cos(theta_grid)
    gain += dipole_gain(theta_grid, phi_grid, frequency, dipole_lengths[1]) * np.exp(1j*phase_reflector)
    
    # Directors (additional elements)
    for i, dir_len in enumerate(dipole_lengths[2:], start=2):
        dir_pos = director_positions[i-2] if i-2 < len(director_positions) else element_spacing*i
        phase_dir = 2*np.pi*(dir_pos/wavelength)*np.cos(theta_grid)
        gain += dipole_gain(theta_grid, phi_grid, frequency, dir_len) * np.exp(1j*phase_dir)
    
    gain_linear = np.abs(gain)**2
    return gain_linear / np.max(gain_linear) if np.max(gain_linear) > 0 else gain_linear

# ---------------------------
# Field Visualization Functions
# ---------------------------
def plot_fields(theta, phi, field, field_type, antenna_type, distance, stride=5):
    """General function for plotting E/H fields"""
    theta_ds, phi_ds = theta[::stride], phi[::stride]
    th_grid, ph_grid = np.meshgrid(theta_ds, phi_ds, indexing='ij')
    
    # Observation positions (sphere of radius 'distance')
    x = distance * np.sin(th_grid) * np.cos(ph_grid)
    y = distance * np.sin(th_grid) * np.sin(ph_grid)
    z = distance * np.cos(th_grid)
    
    # Field components (downsampled)
    fx = field[::stride, ::stride, 0]
    fy = field[::stride, ::stride, 1]
    fz = field[::stride, ::stride, 2]
    
    # Scale vectors for visibility
    mag = np.sqrt(fx**2 + fy**2 + fz**2)
    scale = 0.2 * distance / mag.max() if mag.max() > 0 else 1.0
    fx_scaled, fy_scaled, fz_scaled = fx*scale, fy*scale, fz*scale
    
    # Create plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.quiver(x, y, z, fx_scaled, fy_scaled, fz_scaled, 
              color='red' if field_type == 'E' else 'blue',
              label=f'{field_type}-Field', length=1, normalize=False)
    
    # Reference sphere
    th_sphere, ph_sphere = np.meshgrid(np.linspace(0, np.pi, 100), 
                                      np.linspace(0, 2*np.pi, 100), indexing='ij')
    ax.plot_surface(distance*np.sin(th_sphere)*np.cos(ph_sphere),
                    distance*np.sin(th_sphere)*np.sin(ph_sphere),
                    distance*np.cos(th_sphere),
                    color='gray', alpha=0.2, linewidth=0)
    
    ax.set_xlabel('X (m)'), ax.set_ylabel('Y (m)'), ax.set_zlabel('Z (m)')
    ax.set_title(f'3D {field_type}-Field ({antenna_type}), Distance={distance:.1f}m')
    ax.view_init(elev=30, azim=45)
    plt.legend(), plt.show()

# ---------------------------
# Interactive Simulation Function
# ---------------------------
def interactive_simulation(
    frequency=900e6,
    antenna_type='Isotropic',
    dipole_length=0.5,
    num_elements=3,
    element_spacing=0.1,
    phase_shift=0.0,
    dipole_lengths_key="Standard (Driven: 0.5m, Reflector: 0.55m, 2 Directors)",
    director_positions_key="Breit (0.1m, 0.2m)",
    visualization='Gain',
    distance=1.0
):
    # Angle grids
    theta = np.linspace(0, np.pi, 100)  # 0=up, π=down
    phi = np.linspace(0, 2*np.pi, 100)
    th_grid, ph_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Antenna parameters
    dipole_lengths = YAGI_DIPOLE_LENGTHS[dipole_lengths_key]
    director_pos = DIRECTOR_POSITIONS[director_positions_key]
    
    # Compute gain
    if antenna_type == 'Isotropic':
        gain_linear = isotropic_gain(th_grid, ph_grid)
    elif antenna_type == 'Dipole':
        gain_linear = dipole_gain(th_grid, ph_grid, frequency, dipole_length)
    elif antenna_type == 'Linear Array':
        gain_linear = linear_array_gain(th_grid, ph_grid, frequency, num_elements, element_spacing, phase_shift, dipole_length)
    elif antenna_type == 'Yagi':
        gain_linear = yagi_gain(th_grid, ph_grid, frequency, dipole_lengths, director_pos, element_spacing)
    else:
        gain_linear = isotropic_gain(th_grid, ph_grid)
    
    # Convert to dB
    gain_linear = np.where(gain_linear < 1e-12, 1e-12, gain_linear)  # Avoid log issues
    gain_db = 20 * np.log10(gain_linear)
    gain_db = np.clip(gain_db, -60, 0)
    
    # Visualization logic
    if visualization == 'Gain':
        # Plot gain pattern
        r = np.sqrt(10**(gain_db/20) / 10**(np.max(gain_db)/20))  # Normalize radius
        x = r * np.sin(th_grid) * np.cos(ph_grid)
        y = r * np.sin(th_grid) * np.sin(ph_grid)
        z = r * np.cos(th_grid)
        
        fig = plt.figure(figsize=(18, 8))
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        
        surf = ax3d.plot_surface(x, y, z, facecolors=plt.cm.viridis(gain_db), 
                                 rstride=1, cstride=1, linewidth=0)
        surf.set_clim(-60, 0)
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(-60, 0)), 
                     ax=ax3d, shrink=0.8, label='Gain (dB)')
        
        # 2D cross-section
        idx = np.argmin(abs(phi))
        ax2d.plot(90 - np.degrees(theta), gain_db[:, idx], 'k-', lw=2)
        ax2d.set_xlim(-90, 90), ax2d.set_ylim(-60, 0), ax2d.grid(True, linestyle='--')
        ax2d.set_xlabel('Elevation (deg)'), ax2d.set_ylabel('Gain (dB)')
        
        ax3d.set_title(f'3D Directivity ({antenna_type})'), ax3d.grid(False)
        plt.tight_layout(), plt.show()
    
    else:  # E/H Field visualization
        P_total = 1.0  # Radiated power (W)
        eta = 377.0    # Free space impedance (Ω)
        S_isotropic = P_total / (4 * np.pi * distance**2)  # Isotropic power density
        S = gain_linear * S_isotropic                       # Actual power density
        
        # Compute E/H fields (shape (Nθ, Nφ, 3) for (x,y,z))
        field = np.zeros((th_grid.shape[0], th_grid.shape[1], 3))
        if visualization == 'E-Field':
            if antenna_type == 'Dipole':
                # Eθ field for dipole (converted to Cartesian)
                E_theta = np.sqrt(2 * S / eta)
                et_x = np.cos(th_grid) * np.cos(ph_grid)
                et_y = np.cos(th_grid) * np.sin(ph_grid)
                et_z = -np.sin(th_grid)
                field[..., 0] = E_theta * et_x
                field[..., 1] = E_theta * et_y
                field[..., 2] = E_theta * et_z
            elif antenna_type == 'Isotropic':
                # Radial E field for isotropic
                E_r = np.sqrt(2 * S_isotropic / eta)
                field[..., 0] = E_r * np.sin(th_grid) * np.cos(ph_grid)
                field[..., 1] = E_r * np.sin(th_grid) * np.sin(ph_grid)
                field[..., 2] = E_r * np.cos(th_grid)
            else:
                print(f"E-Feld für {antenna_type} nicht implementiert")
        
        elif visualization == 'H-Field':
            if antenna_type == 'Dipole':
                # Hφ field (Eθ / eta)
                E_theta = np.sqrt(2 * S / eta)
                H_phi = E_theta / eta
                ep_x = -np.sin(ph_grid)
                ep_y = np.cos(ph_grid)
                field[..., 0] = H_phi * ep_x
                field[..., 1] = H_phi * ep_y
            elif antenna_type == 'Isotropic':
                # Azimuthal H field (E_r / eta)
                E_r = np.sqrt(2 * S_isotropic / eta)
                H_phi = E_r / eta
                ep_x = -np.sin(ph_grid)
                ep_y = np.cos(ph_grid)
                field[..., 0] = H_phi * ep_x
                field[..., 1] = H_phi * ep_y
            else:
                print(f"H-Feld für {antenna_type} nicht implementiert")
        
        # Plot fields
        plot_fields(theta, phi, field, visualization[0], antenna_type, distance)

# ---------------------------
# Interactive Widgets Setup
# ---------------------------
interact(
    interactive_simulation,
    frequency=FloatSlider(min=1e6, max=6e9, step=1e6, value=900e6, description='Frequency (Hz):'),
    antenna_type=Dropdown(options=['Isotropic', 'Dipole', 'Linear Array', 'Yagi'], value='Isotropic', description='Antenna Type:'),
    dipole_length=FloatSlider(min=0.1, max=2.0, step=0.05, value=0.5, description='Dipole Length (m):'),
    num_elements=IntSlider(min=1, max=10, value=3, description='Array Elements (N):'),
    element_spacing=FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='Element Spacing (m):'),
    phase_shift=FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0.0, description='Phase Shift (rad):'),
    dipole_lengths_key=Dropdown(options=list(YAGI_DIPOLE_LENGTHS.keys()), description='Yagi Lengths:'),
    director_positions_key=Dropdown(options=list(DIRECTOR_POSITIONS.keys()), description='Yagi Directors:'),
    visualization=Dropdown(options=['Gain', 'E-Field', 'H-Field'], value='Gain', description='Visualization:'),
    distance=FloatSlider(min=0.1, max=10.0, step=0.1, value=1.0, description='Distance (m):')
);