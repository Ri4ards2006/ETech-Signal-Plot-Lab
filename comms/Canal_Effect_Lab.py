import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, convolve

# --------------------------
# Signalparameter einstellen
# --------------------------
N_bits = 1000          # Anzahl zu sendender Bits
R_b = 1e6             # Bitrate (bps)
alpha = 0.2           # Roll-off-Faktor für Raised-Cosine-Filter
Fs = 10 * R_b         # Abtastrate (Samples/s), 10× Überabtastung
modulation = 'BPSK'   # Modulation (nur BPSK implementiert)

# --------------------------
# Channelparameter einstellen
# --------------------------
SNR_dB = 20           # SNR für AWGN (dB)
N_reflections = 3    # Anzahl Multipath-Reflexionen (0 = deaktiviert)
tau_max = 1e-6        # Maximale Zeitverschiebung (s) für Multipath
f_d = 0               # Dopplerfrequenz (Hz) (0 = deaktiviert)
fading_type = 'None'  # Fadingtyp: 'Rayleigh', 'Rician' oder 'None'
K = 10                # Rician-Faktor (nur bei 'Rician' relevant)

# --------------------------
# Carrier-Parameter (für Doppler)
# --------------------------
f_c = 2.4e9           # Trägerfrequenz (Hz)
c = 3e8               # Lichtgeschwindigkeit (m/s)

# --------------------------
# 1. Signalgenerierung (BPSK)
# --------------------------
bits = np.random.randint(0, 2, N_bits)  # Zufällige Bits generieren
if modulation == 'BPSK':
    symbols = 2 * bits - 1  # Bits zu Symbolen konvertieren (0→-1, 1→1)
else:
    raise ValueError("Nur BPSK wird unterstützt.")

# --------------------------
# 2. Überabtastung und Pulse-Shaping (Raised Cosine)
# --------------------------
samples_per_symbol = int(Fs // R_b)  # Samples pro Symbol (10 bei Fs=10e6, R_b=1e6)
L = N_bits * samples_per_symbol     # Gesamtlength des Basisband Signals (überabgetastet)

# Überabgetastetes Signal erstellen (Symbole eingebettet in Nullen)
baseband_upsampled = np.zeros(L, dtype=complex)
baseband_upsampled[::samples_per_symbol] = symbols  # Symbole alle samples_per_symbol Samples

# Raised-Cosine-Filter designen
filter_span = 10  # Filterlänge in Symbolperioden
num_taps = filter_span * samples_per_symbol  # Anzahl Filtertaps
cutoff_freq = (1 + alpha) * R_b / 2  # Cut-off-Frequenz mit Roll-off
normalized_cutoff = cutoff_freq / (Fs / 2)  # Normalisierte Cut-off-Frequenz (0-1)
rc_filter = firwin(num_taps, normalized_cutoff, window=('kaiser', 8), fs=Fs)

# Pulse-Shaping anwenden (Filterung)
baseband_clean = lfilter(rc_filter, 1.0, baseband_upsampled)

# Zeitvektor für Basisband-Signal (ursprünglich)
t_baseband = np.arange(len(baseband_clean)) / Fs

# --------------------------
# 3. Doppler-Effekt anwenden (falls aktiviert)
# --------------------------
if f_d != 0:
    doppler_phase = np.exp(1j * 2 * np.pi * f_d * t_baseband)  # Phase für Doppler
    baseband_doppler = baseband_clean * doppler_phase
else:
    baseband_doppler = baseband_clean.copy()

# --------------------------
# 4. Multipath-Fading anwenden (falls Reflexionen > 0)
# --------------------------
baseband_multipath = baseband_doppler.copy()  # Initiiere multipath-signal
if N_reflections > 0:
    N_paths = N_reflections + 1  # Gerade Path (1) + Reflexionen (N_reflections)
    delay_samples = np.zeros(N_paths, dtype=int)
    gains = np.zeros(N_paths, dtype=complex)
    
    # Gerade Path (kein Delay, volle Gewichtung)
    delay_samples[0] = 0
    gains[0] = 1.0
    
    # Zufällige Reflexionen generieren (einzigartige Delays)
    for i in range(1, N_paths):
        while True:
            tau = np.random.uniform(0, tau_max)  # Zufälliger Delay in Sekunden
            d = int(round(tau * Fs))            # Konvertiere zu Samples
            if d > 0 and d not in delay_samples[:i]:  # Unique und positiv
                delay_samples[i] = d
                # Rayleigh-Gewichtung (Amplitude und Phase)
                gain_mag = np.random.rayleigh(1.0)     # Amplitude (Rayleigh-verteilt)
                gain_phase = np.random.uniform(0, 2 * np.pi)  # Phase (uniform)
                gains[i] = gain_mag * np.exp(1j * gain_phase)
                break
    
    # Gewichte normalisieren, um Gesamtleistung zu erhalten
    sum_gains_sq = np.sum(np.abs(gains) ** 2)
    if sum_gains_sq > 0:
        gains /= np.sqrt(sum_gains_sq)
    
    # Kanalimpulsantwort (h) erstellen
    max_delay = np.max(delay_samples)
    h = np.zeros(max_delay + 1, dtype=complex)
    for i in range(N_paths):
        d = delay_samples[i]
        h[d] += gains[i]
    
    # Multipath anwenden (Faltung)
    baseband_multipath = convolve(baseband_doppler, h)

# --------------------------
# 5. Rayleigh/Rician-Fading anwenden (falls aktiviert)
# --------------------------
if fading_type != 'None':
    if fading_type == 'Rayleigh':
        # Rayleigh-Fading: komplexes Gauß-Rauschen (Mittelwert |h|²=1)
        h_fading = (np.random.normal(0, 1/np.sqrt(2)) + 
                    1j * np.random.normal(0, 1/np.sqrt(2)))
    elif fading_type == 'Rician':
        # Rician-Fading: Line-of-Sight + Rayleigh (Mittelwert |h|²=1)
        theta = np.random.uniform(0, 2 * np.pi)  # LOS-Phase
        z = (np.random.normal(0, 1/np.sqrt(2)) + 
             1j * np.random.normal(0, 1/np.sqrt(2)))  # Zufälliger Teil
        h_fading = np.sqrt(K/(K+1)) * np.exp(1j*theta) + (1/np.sqrt(K+1)) * z
    else:
        raise ValueError("Ungültiger Fadingtyp.")
    
    baseband_faded = baseband_multipath * h_fading
else:
    baseband_faded = baseband_multipath.copy()

# --------------------------
# 6. AWGN hinzufügen
# --------------------------
signal_power = np.mean(np.abs(baseband_faded) ** 2)  # Signalleistung (ohne Rauschen)
noise_power = signal_power / (10 ** (SNR_dB / 10))  # Rauschleistung basierend auf SNR

# Komplexes Gauss-Rauschen erzeugen (Real/Imag Teil unabhängig)
noise = np.sqrt(noise_power / 2) * (
    np.random.normal(size=len(baseband_faded)) + 
    1j * np.random.normal(size=len(baseband_faded))
)
baseband_noisy = baseband_faded + noise  # Signal mit Rauschen

# Zeitvektor für verschmutztes Signal (prolongiert durch Faltung)
t_noisy = np.arange(len(baseband_noisy)) / Fs

# --------------------------
# 7. BER berechnen (simplifiziert, ohne Gleicheitung)
# --------------------------
# Originalsymbole (abgetastet)
original_symbols = symbols.copy()

# Empfangensymbole extrahieren (stabile Phase nach Faltung)
if len(baseband_noisy) < (L + max_delay):
    baseband_noisy_padded = np.pad(baseband_noisy, (0, (L + max_delay) - len(baseband_noisy)), mode='constant')
else:
    baseband_noisy_padded = baseband_noisy.copy()

baseband_noisy_steady = baseband_noisy_padded[max_delay:]  # Ignoriere initiale Faltungs-Artefakte
received_symbols = baseband_noisy_steady[::samples_per_symbol]  # Abtasten auf Symbolrate

# Demodulation (BPSK: Realteil >0 → 1, sonst 0)
received_bits = (np.real(received_symbols) >= 0).astype(int)

# BER berechnen
BER = np.mean(received_bits != bits)
print(f"BER für SNR={SNR_dB} dB: {BER:.4f}")

# --------------------------
# 8. Visualisierung der Ergebnisse
# --------------------------

def plot_time_domain(signal, t, title, label='Signal'):
    """Zeitbereichs-Plot"""
    plt.figure(figsize=(12, 5))
    plt.plot(t, np.real(signal), label=f'Real {label}', alpha=0.7)
    if np.any(np.imag(signal)):
        plt.plot(t, np.imag(signal), label=f'Imag {label}', linestyle='--', alpha=0.5)
    plt.title(title)
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_freq_domain(signal, Fs, title):
    """Frequenzbereichs-Plot"""
    N = len(signal)
    Y = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, 1/Fs)
    plt.figure(figsize=(12, 5))
    plt.plot(freq, np.abs(Y), label='Magnitude')
    plt.title(title)
    plt.xlabel('Frequenz (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(-Fs/2, Fs/2)
    plt.grid(True)
    plt.show()

def plot_eye_diagram(signal, samples_per_symbol, title):
    """Eye-Diagramm für digitale Signale"""
    L = len(signal)
    num_symbols = L // samples_per_symbol
    if num_symbols == 0:
        return  # Keine Symbole zum Plotten
    signal_trunc = signal[:num_symbols * samples_per_symbol]
    signal_reshape = signal_trunc.reshape(num_symbols, samples_per_symbol)
    t_eye = np.arange(samples_per_symbol) / Fs  # Zeit innerhalb eines Symbols
    
    plt.figure(figsize=(12, 5))
    for i in range(num_symbols):
        plt.plot(t_eye, np.real(signal_reshape[i, :]), 'b', alpha=0.1)
    plt.title(title)
    plt.xlabel('Zeit innerhalb des Symbols (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Originalsignal (Zeit/Frequenz)
plot_time_domain(baseband_clean, t_baseband, 'Originalsignal (Zeitbereich)', 'Original')
plot_freq_domain(baseband_clean, Fs, 'Originalsignal (Frequenzbereich)')

# Signal nach Kanalübertragung (Zeit/Frequenz)
plot_time_domain(baseband_noisy, t_noisy, 'Signal nach Kanal (Zeitbereich)', 'Nach Channel')
plot_freq_domain(baseband_noisy, Fs, 'Signal nach Kanal (Frequenzbereich)')

# Eye-Diagramme (Original vs. nach Channel)
plot_eye_diagram(baseband_clean, samples_per_symbol, 'Eye-Diagramm - Originalsignal')
plot_eye_diagram(baseband_noisy, samples_per_symbol, 'Eye-Diagramm - Signal nach Kanal')

# --------------------------
# Optional: BER vs. SNR Plot (falls SNR variiert werden soll)
# --------------------------
if False:  # Setzen Sie auf True, um den Plot zu generieren
    SNR_dB_values = np.arange(0, 30, 5)  # Prüfe SNR von 0 bis 30 dB
    BER_values = []
    
    for snr in SNR_dB_values:
        # Neues Rauschen für aktuellen SNR generieren
        noise_power_current = signal_power / (10 ** (snr / 10))
        noise_current = np.sqrt(noise_power_current / 2) * (
            np.random.normal(size=len(baseband_faded)) + 
            1j * np.random.normal(size=len(baseband_faded))
        )
        baseband_noisy_current = baseband_faded + noise_current
        
        # Empfangensymbole extrahieren
        if len(baseband_noisy_current) < (L + max_delay):
            padded = np.pad(baseband_noisy_current, (0, (L + max_delay) - len(baseband_noisy_current)), mode='constant')
        else:
            padded = baseband_noisy_current.copy()
        steady = padded[max_delay:]
        received_symbols_current = steady[::samples_per_symbol]
        received_bits_current = (np.real(received_symbols_current) >= 0).astype(int)
        BER = np.mean(received_bits_current != bits)
        BER_values.append(BER)
    
    # Plot BER vs. SNR
    plt.figure(figsize=(12, 6))
    plt.semilogy(SNR_dB_values, BER_values, 'ro-', linewidth=2)
    plt.title('BER vs. SNR (Drahtloschannel)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Biterrorrate (BER)')
    plt.grid(True)
    plt.show()