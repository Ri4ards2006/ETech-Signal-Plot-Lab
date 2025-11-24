# --- NEUE FUNKTIONEN (Impedanz-Transformationen) ---

def series_L_transform(Z_in, f, L_L):
    """
    Transformiert die Impedanz Z_in durch Hinzufügen eines Serien-Induktors L_L.
    Z_total = Z_in + j*ω*L_L.
    """
    omega = 2 * np.pi * f
    Z_L = 1j * omega * L_L
    return Z_in + Z_L

def shunt_C_transform(Z_in, f, C_C):
    """
    Transformiert die Impedanz Z_in durch Hinzufügen eines Shunt-Kondensators C_C.
    Verwendet die Formel für parallele Impedanz: Z_total = (Z_in * Z_C) / (Z_in + Z_C).
    """
    if C_C <= 0:
        return Z_in  # Kein Effekt, wenn C ungültig
        
    omega = 2 * np.pi * f
    Z_C = 1/(1j * omega * C_C)  # Kapazitive Impedanz: 1/(jωC)
    
    # Parallele Impedanz
    return (Z_in * Z_C) / (Z_in + Z_C)

# --- ZENTRALER DISPATCHER ---

def compute_total_impedance(f, Z_ant_func, network_type, L_L, C_C, Z0):
    """
    Berechnet die Gesamtimpedanz nach Anwendung des gewählten Matching-Netzwerks.
    
    Args:
        f (np.ndarray): Frequenz-Array (Hz).
        Z_ant_func (callable): Funktion zur Berechnung der Antennenimpedanz.
        network_type (str): Gewählte Netzwerk-Konfiguration ('None', 'Series L', 'Shunt C').
        L_L (float): Induktorwert (H).
        C_C (float): Kondensatorwert (F).
        Z0 (float): Referenzimpedanz (Ω).

    Returns:
        np.ndarray: Gesamtimpedanz (Ω).
    """
    Z_ant = Z_ant_func(f)  # Roh-Antennenimpedanz
    
    if network_type == 'None':
        Z_total = Z_ant
    elif network_type == 'Series L':
        Z_total = series_L_transform(Z_ant, f, L_L)
    elif network_type == 'Shunt C':
        Z_total = shunt_C_transform(Z_ant, f, C_C)
    else:
        Z_total = Z_ant  # Fallback
        
    return Z_total

# --- UPDATE DER MAIN SIMULATIONSFUNKTION (V1-Code muss angepasst werden) ---

def run_basic_simulation_v2(f_min_mhz=5, f_max_mhz=20, R_ant=50, L_ant=1e-6, C_ant=1e-9, Z0=50, 
                            network_type='None', L_L=0.5e-6, C_C=0.5e-9):
    # ... (Rest des Setups wie in V1) ...
    num_f = 500
    f_min_hz = f_min_mhz * 1e6
    f_max_hz = f_max_mhz * 1e6
    f = np.linspace(f_min_hz, f_max_hz, num_f)
    
    # 1. Compute Impedances
    # Lambda-Funktion für den Z_ant_func-Parameter
    Z_ant_model = lambda f: compute_antenna_impedance(f, R_ant, L_ant, C_ant, X0=0)
    
    # Gesamtimpedanz berechnen
    Z_total = compute_total_impedance(
        f, 
        Z_ant_model, 
        network_type, 
        L_L, C_C, 
        Z0
    )
    
    # Roh-Antennenimpedanz (für Vergleich)
    Z_ant = Z_ant_model(f) 
    
    # 2. Compute Metrics
    gamma_total = compute_gamma(Z_total, Z0)
    return_loss_total = compute_return_loss(gamma_total)
    
    gamma_ant = compute_gamma(Z_ant, Z0)
    return_loss_ant = compute_return_loss(gamma_ant)
    
    # 3. Plotting (Jetzt mit Vergleich von Antenne vs. Total)
    plt.figure(figsize=(10, 6))
    plt.plot(f/1e6, return_loss_total, color='red', linewidth=2, label=f'Total RL ({network_type})')
    plt.plot(f/1e6, return_loss_ant, color='blue', linestyle='--', linewidth=1, label='Antenna RL (No Matching)')

    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Return Loss (dB)')
    plt.title(f'V2 - Commit 1: Return Loss Spectrum with {network_type} Matching')
    plt.ylim(-60, 0)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.show()

# --- RUN V2 COMMIT 1 (Beispiel) ---
if __name__ == '__main__':
    # Beispiel 1: Nur Antenne (Sollte wie V1 aussehen)
    print("--- Running V2 Commit 1: No Matching ---")
    run_basic_simulation_v2(network_type='None')
    
    # Beispiel 2: Mit Serien-Induktor
    print("--- Running V2 Commit 1: Series L Matching ---")
    run_basic_simulation_v2(network_type='Series L', L_L=2.0e-6)