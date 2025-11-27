import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, Checkbox
from scipy.signal import hilbert, butter, filtfilt
import sounddevice as sd  # Optional, für Audio-Vorschau

# --- Einstellungen ---
plt.style.use('seaborn')  # Schönere Plot-Aussehen
SAMPLE_RATE = 44100  # Audio-Samplerate (Hz), irrelevant für Plots aber für Audio
DURATION = 1.0  # Simulationsdauer (s)
FREQ_CARRIER_MIN = 1e3   # Min Trägerfrequenz (Hz)
FREQ_CARRIER_MAX = 1e6  # Max Trägerfrequenz (Hz)
MODULATION_INDEX_MAX = 2  # Max Modulationsindex (AM: m≤1, FM: β kann höher sein)
SNR_MIN = -20  # Min SNR (dB)
SNR_MAX = 30   # Max SNR (dB)

# --- Hilfsfunktionen ---
def generate_signal(t, signal_type, freq_signal=1000, amplitude=1.0):
    """
    Generiere Eingangssignal (Modulationsbasis)
    Args:
        t (np.ndarray): Zeitvektor (s)
        signal_type (str): 'sinus' oder 'binary' (Binärsequenz)
        freq_signal (float): Signalfrequenz (Hz) für Sinus oder Bitrate (bps) für Binär
        amplitude (float): Signalamplitude (Ω)
    Returns:
        np.ndarray: Eingangssignal (s(t))
    """
    if signal_type == 'sinus':
        return amplitude * np.sin(2 * np.pi * freq_signal * t)
    elif signal_type == 'binary':
        # Binärsequenz mit Bitrate = freq_signal (bps)
        num_bits = int(freq_signal * DURATION)
        bits = np.random.randint(0, 2, num_bits)
        # Konvertiere zu Bipolar (0→amplitude, 1→-amplitude)
        return np.where(bits == 1, -amplitude, amplitude)
    else:
        return np.zeros_like(t)

def add_awgn(signal, snr_dB):
    """
    Füge AWGN (Additiver Weißes Gauß-Noise) zum Signal hinzu
    Args:
        signal (np.ndarray): Eingangssignal
        snr_dB (float): SNR in dB (≤-100 → kein Rauschen)
    Returns:
        np.ndarray: Signal mit Rauschen
    """
    if snr_dB <= -100:  # Kein Rauschen bei extrem niedrigem SNR
        return signal
    
    signal_power = np.mean(signal**2)  # Signal-Power (RMS²)
    noise_power = signal_power / (10 ** (snr_dB / 10))  # Rausch-Power
    noise_amplitude = np.sqrt(noise_power)  # Rausch-Amplitude (Std-Abweichung)
    noise = noise_amplitude * np.random.normal(0, 1, len(signal))
    return signal + noise

def demodulate_am(modulated_signal, carrier_freq, sample_rate):
    """Envelopendemodulation für AM-Signal"""
    analytic_signal = hilbert(modulated_signal)
    envelope = np.abs(analytic_signal)
    cutoff = carrier_freq / 2  # Filtere Carrier und höhere Frequenzen
    b, a = butter(4, cutoff, fs=sample_rate, output='ba')
    demodulated = filtfilt(b, a, envelope)
    return demodulated

def demodulate_fm(modulated_signal, carrier_freq, sample_rate):
    """Phasendifferenz-Demodulation für FM-Signal"""
    analytic_signal = hilbert(modulated_signal)
    phase = np.angle(analytic_signal)
    # Phasendifferenz → Frequenzabweichung (Δf)
    freq_deviation = np.diff(phase) / (2 * np.pi) * sample_rate
    # Entferne DC-Offset (ausgleichen)
    freq_deviation -= np.mean(freq_deviation)
    return freq_deviation

def demodulate_digital(modulated_signal, carrier_freq, sample_rate, symbol_rate, t, amplitude_carrier, amplitude_signal):
    """Demodulation für digitale Signale (BPSK/QPSK) inkl. I/Q-Kanal-Bearbeitung"""
    # I-Kanal: Demodulation mit cos(2πf_c t), LPF
    i_demod = modulated_signal * np.cos(2 * np.pi * carrier_freq * t)
    # Q-Kanal: Demodulation mit sin(2πf_c t), LPF (90° Phasenverschiebung)
    q_demod = modulated_signal * np.sin(2 * np.pi * carrier_freq * t)
    
    # Tiefpassfilter mit Cut-off = symbol_rate/2 (entspricht Symbolzeit/Halb)
    cutoff = symbol_rate / 2
    cutoff = max(cutoff, 1e-3)  # Vermeide Division durch 0
    b, a = butter(4, cutoff, fs=sample_rate, output='ba')
    filtered_i = filtfilt(b, a, i_demod)
    filtered_q = filtfilt(b, a, q_demod)
    
    # Sample an Symbolzeiten (jede 1/symbol_rate)
    symbol_times = np.arange(0, DURATION, 1/symbol_rate)
    samples_i = np.interp(symbol_times, t, filtered_i)
    samples_q = np.interp(symbol_times, t, filtered_q)
    
    # Erwartete Amplitude der demodulierten Symbole (nach Skalierung)
    expected_amp = 0.5 * amplitude_carrier * amplitude_signal  # Skalierung durch cos²/sin²
    
    # Bitrekuperation für BPSK/QPSK via naivem Min-Distance-Algorithmus
    bits_demod = []
    for i in range(len(samples_i)):
        # Akteure I/Q-Werte
        i_val = samples_i[i]
        q_val = samples_q[i]
        
        # Ideal Symbole (I, Q) mit erwarteter Amplitude
        ideal_symbols = [
            (expected_amp, 0),    # (0,0) → BPSK: 1, QPSK: 1
            (0, expected_amp),   # (0,1) → QPSK: j
            (-expected_amp, 0),  # (1,0) → BPSK: -1, QPSK: -1
            (0, -expected_amp)   # (1,1) → QPSK: -j
        ]
        
        # Quadratischer Abstand zu jedem Ideal-Symbol
        d_sq = [
            (i_val - sym_i)**2 + (q_val - sym_q)**2 
            for sym_i, sym_q in ideal_symbols
        ]
        min_idx = np.argmin(d_sq)
        
        # Mappe Index zu Bits (b0, b1)
        if min_idx == 0:
            b0, b1 = 0, 0
        elif min_idx == 1:
            b0, b1 = 0, 1
        elif min_idx == 2:
            b0, b1 = 1, 0
        else:  # min_idx ==3
            b0, b1 = 1, 1
        
        bits_demod.extend([b0, b1])  # Füge beide Bits hinzu (auch für BPSK)
    
    return np.array(bits_demod), samples_i, samples_q

def plot_eye_diagram(ax, samples_i, samples_q, symbol_rate, modulation_type):
    """Eye-Diagramm für digitale Signale (BPSK/QPSK)"""
    ax.clear()
    dt_symbol = 1 / symbol_rate  # Symboldauer (s)
    num_samples_per_symbol = int(SAMPLE_RATE * dt_symbol)  # Samples pro Symbol
    
    # Zeitpunkte innerhalb eines Symbols (0 bis dt_symbol)
    symbol_t = np.linspace(0, dt_symbol, num_samples_per_symbol)
    
    # Zeige Eye-Diagramm für I/Kanal (BPSK) oder I/Q (QPSK)
    if modulation_type == 'BPSK':
        # BPSK: Q-Kanal wird ignoriert (nur I)
        signal_samples = samples_i
        ax.set_title('Eye-Diagramm (I-Kanal)')
    else:  # QPSK
        # Kombiniere I und Q für übersichtliches Plot (optional)
        signal_samples = np.concatenate([samples_i, samples_q])
        symbol_t_q = symbol_t + dt_symbol  # Q-Kanal nach I-Kanal plottten
        all_symbol_t = np.concatenate([symbol_t, symbol_t_q])
        ax.set_title('Eye-Diagramm (I/Kanal + Q-Kanal)')
        # Plotte Q-Kanal in anderer Farbe
        ax.plot(all_symbol_t, signal_samples, 'r', alpha=0.1)
        # Zoom auf I-Kanal (optional)
        ax.set_xlim(0, dt_symbol*2)
    
    # Plotte I-Kanal
    ax.plot(symbol_t, samples_i[:num_samples_per_symbol], 'b', alpha=0.1)
    ax.set_xlabel('Zeit innerhalb des Symbols (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--')
    ax.set_xlim(0, dt_symbol)

def plot_spectrum(ax, signal, sample_rate):
    """Frequenzgang (FFT) des Signals"""
    ax.clear()
    n = len(signal)
    freq = np.fft.fftfreq(n, 1/sample_rate)[:n//2]  # positive Frequenzen
    fft_vals = np.abs(np.fft.fft(signal))[:n//2]
    # Normalisiere FFT auf Max-Amplitude
    fft_vals = fft_vals / np.max(fft_vals) * np.max(np.abs(signal))
    ax.plot(freq, 20 * np.log10(fft_vals + 1e-10), color='r')  # dB-Skala
    ax.set_title('Spektrum (FFT)')
    ax.set_xlabel('Frequenz (Hz)')
    ax.set_ylabel('Amplitude (dB)')
    ax.grid(True, linestyle='--')
    ax.set_xlim(0, sample_rate/2)  # Nyquist-Grenze

def play_audio(signal, sample_rate):
    """Optional: Audio-Vorschau des Signals (mit SoundDevice)"""
    if not np.isnan(signal).any():
        sd.play(signal, sample_rate, blocking=True)

# --- Interaktive Simulation ---
def modulation_toolkit(
    modulation_type='AM',
    signal_type='sinus',
    carrier_freq=10000,  # Hz (Trägerfrequenz)
    signal_freq=1000,   # Hz (Sinus-Frequenz) oder bps (Bitrate für Binär)
    modulation_index=0.5,
    snr_dB=0,
    amplitude_signal=1.0,  # Amplitude des Baseband-Signals
    amplitude_carrier=2.0, # Amplitude des Trägers
    symbol_rate=5000,      # Hz (Symbolrate, nur für digitale Modulationen)
    play_audio_check=False  # Checkbox-Wert für Audio-Vorschau
):
    # Zeitvektor generieren (fixe Länge)
    t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
    
    # 1. Eingangssignal generieren
    s = generate_signal(t, signal_type, signal_freq, amplitude_signal)
    
    # 2. Trägersignal generieren (cos für digitale, sin für analoge Modulationen)
    if modulation_type in ['BPSK', 'QPSK']:
        carrier = amplitude_carrier * np.cos(2 * np.pi * carrier_freq * t)
    else:
        carrier = amplitude_carrier * np.sin(2 * np.pi * carrier_freq * t)
    
    # 3. Modulieren
    if modulation_type == 'AM':
        # AM-DSB: s_mod = (1 + m*s/A_s) * carrier (m=Modulationsindex)
        m = modulation_index
        s_mod = (1 + m * s / (amplitude_signal + 1e-10)) * carrier  # +1e-10 vermeidet /0
    elif modulation_type == 'FM':
        # FM: Δf = β*s (Frequenzabweichung), Phase=∫Δf dt
        beta = modulation_index  # Modulationsindex β
        delta_f = beta * s
        phase = 2 * np.pi * np.cumsum(delta_f) * (DURATION / len(delta_f))  # Integriere über Zeit
        s_mod = amplitude_carrier * np.sin(2 * np.pi * carrier_freq * t + phase)
    elif modulation_type == 'BPSK':
        # BPSK: 1 Bit/Symbol, I-Kanal mit Bipolar-Signal
        # Bitrate R_b = symbol_rate (1 Bit/Symbol)
        R_b = symbol_rate  # Bitrate
        num_bits = int(R_b * DURATION)
        # Generiere Bipolar-Signal (direkt, da generate_signal für Binär bereits verwendet)
        bits = np.random.randint(0, 2, num_bits)
        s_symbols = np.where(bits == 1, amplitude_signal, -amplitude_signal)
        # Pulse-Länge (Samples pro Symbol)
        pulse_length = int(SAMPLE_RATE / symbol_rate)
        # Upsample: jedes Symbol pulse_length-mal wiederholen
        baseband = np.repeat(s_symbols, pulse_length)
        baseband = baseband[:len(t)]  # Trim to t-Länge
        s_mod = baseband * carrier  # Moduliere mit Träger (cos)
    elif modulation_type == 'QPSK':
        # QPSK: 2 Bits/Symbol, I/Q-Kanäle mit Symbolen
        # Bitrate R_b = 2 * symbol_rate
        R_b = 2 * symbol_rate
        num_bits = int(R_b * DURATION)
        bits = np.random.randint(0, 2, num_bits)
        # Padd mit 0, falls ungerade Anzahl Bits
        if num_bits % 2 != 0:
            bits = np.pad(bits, (0,1), mode='constant')
            num_bits += 1
        # Gruppiere Bits zu Symbolen (2 Bits/Symbol)
        symbols = []
        for i in range(0, num_bits, 2):
            b0, b1 = bits[i], bits[i+1]
            # Standard QPSK-Mapping (I + jQ)
            if (b0, b1) == (0, 0):
                sym = 1
            elif (b0, b1) == (0, 1):
                sym = 1j
            elif (b0, b1) == (1, 0):
                sym = -1
            else:  # (1,1)
                sym = -1j
            symbols.append(sym * amplitude_signal)  # Skalieren mit Signal-Amplitude
        symbols = np.array(symbols)
        # Pulse-Länge (Samples pro Symbol)
        pulse_length = int(SAMPLE_RATE / symbol_rate)
        # Upsample I und Q-Komponenten
        baseband_i = np.repeat(symbols.real, pulse_length)
        baseband_q = np.repeat(symbols.imag, pulse_length)
        # Trim to t-Länge
        baseband_i = baseband_i[:len(t)]
        baseband_q = baseband_q[:len(t)]
        # Moduliere I und Q mit Träger (cos/sin)
        carrier_i = amplitude_carrier * np.cos(2 * np.pi * carrier_freq * t)
        carrier_q = amplitude_carrier * np.sin(2 * np.pi * carrier_freq * t)
        s_mod_i = baseband_i * carrier_i
        s_mod_q = baseband_q * carrier_q
        s_mod = s_mod_i + s_mod_q  # Summe I/Q
    else:
        s_mod = carrier  # Fallback
    
    # 4. Rauschen hinzufügen
    s_mod_noisy = add_awgn(s_mod, snr_dB)
    
    # 5. Demodulieren
    if modulation_type == 'AM':
        s_demod = demodulate_am(s_mod_noisy, carrier_freq, SAMPLE_RATE)
        # Korrektur: AM-Demodulation liefert Baseband-Amplitude, nicht original signal
        s_demod = s_demod / np.max(s_demod) * amplitude_signal  # Skalieren auf Original-Amplitude
    elif modulation_type == 'FM':
        s_demod = demodulate_fm(s_mod_noisy, carrier_freq, SAMPLE_RATE)
        # FM-Demodulation liefert Frequenzabweichung → konvertiere zu Originalsignal
        s_demod = s_demod / np.max(np.abs(s_demod)) * amplitude_signal  # Skalieren
    else:  #_digitale (BPSK/QPSK)
        # Demoduliere und erhält I/Q-Samples + Bits
        bits_demod, samples_i, samples_q = demodulate_digital(
            s_mod_noisy, carrier_freq, SAMPLE_RATE, symbol_rate, t, 
            amplitude_carrier, amplitude_signal
        )
        # Extrahiere I-Kanal-Samples für BPSK oder kombiniere für QPSK
        if modulation_type == 'BPSK':
            s_demod = samples_i
            s_baseband = np.repeat(s_symbols, pulse_length)[:len(bits_demod)]  # Original Baseband für Plot
        else:  # QPSK
            s_demod = bits_demod
            # Original Bits (ohne Padding) für Plot
            s_baseband = bits[:len(bits_demod)]
        # Skalieren der Samples (optional, für bessere Sichtbarkeit)
        samples_i_scaled = samples_i / expected_amp * amplitude_signal if (modulation_type == 'QPSK' and expected_amp !=0) else samples_i
        # (expected_amp wird in demodulate_digital verwendet, aber hier nicht definiert → adjustiere)
    
    # --- Plots ---
    plt.close('all')  # Alte Plots löschen
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Originalsignal (Zeitbereich)
    ax1.plot(t, s, color='b', lw=1.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Eingangssignal ({signal_type})')
    ax1.grid(True, linestyle='--')
    
    # Plot 2: Moduliertes Signal (Zeitbereich)
    ax2.plot(t, s_mod, color='g', lw=1.5, label='Ohne Rauschen')
    ax2.plot(t, s_mod_noisy, color='r', lw=0.8, alpha=0.7, label=f'Mit Rauschen (SNR={snr_dB:.1f} dB)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Moduliertes Signal ({modulation_type})')
    ax2.legend()
    ax2.grid(True, linestyle='--')
    
    # Plot 3: Frequenzgang
    plot_spectrum(ax3, s_mod_noisy, SAMPLE_RATE)
    ax3.set_xlim(0, 1.2 * carrier_freq)  # Zoom auf Trägerbereich
    
    # Plot 4: Demoduliertes Signal / Eye-Diagramm
    if modulation_type in ['AM', 'FM']:
        # Analog-Demodulation
        ax4.plot(t, s_demod, color='m', lw=1.5, label='Demoduliertes Signal')
        ax4.plot(t, s, color='b', lw=0.8, alpha=0.3, label='Originalsignal')
        ax4.set_ylabel('Amplitude')
        ax4.set_title(f'Demodulation ({modulation_type})')
        ax4.legend()
        ax4.set_ylim(-1.2 * amplitude_signal, 1.2 * amplitude_signal)
    else:
        # Digitale-Demodulation → Eye-Diagramm
        plot_eye_diagram(ax4, samples_i, samples_q, symbol_rate, modulation_type)
        # Optional: Plot originaler Baseband-Bits (ohne Rauschen)
        if signal_type == 'binary':
            baseband_t = np.linspace(0, DURATION, len(s_baseband))
            ax4.plot(baseband_t, s_baseband * amplitude_signal, 'k--', lw=0.5, 
                     label='Original Baseband', alpha=0.5)
            ax4.legend()
    ax4.set_xlabel('Zeit (s)')
    ax4.grid(True, linestyle='--')
    
    plt.suptitle(f'Modulationsvergleich: {modulation_type} | Carrier={carrier_freq} Hz | Signal={signal_freq} Hz', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Optional: Audio-Vorschau (nur wenn Signal nicht zu hochfrequent und SNR ok)
    if carrier_freq < SAMPLE_RATE/2 and snr_dB > -50 and play_audio_check:
        play_audio(s_mod_noisy, SAMPLE_RATE)

# --- Widgets definieren ---
# Checkbox für Audio-Vorschau
play_audio_check = Checkbox(value=False, description='Play Audio')

interact(
    modulation_toolkit,
    modulation_type=Dropdown(
        options=['AM', 'FM', 'BPSK', 'QPSK'],
        value='AM',
        description='Modulationstyp:'
    ),
    signal_type=Dropdown(
        options=['sinus', 'binary'],
        value='sinus',
        description='Eingangstyp:'
    ),
    carrier_freq=FloatSlider(
        min=FREQ_CARRIER_MIN, max=FREQ_CARRIER_MAX, step=100, 
        value=10000, description='Trägerfrequenz (Hz):'
    ),
    signal_freq=FloatSlider(
        min=10, max=1e4, step=10, 
        value=1000, description='Signalfrequenz/Bitrate (Hz/bps):'
    ),
    modulation_index=FloatSlider(
        min=0, max=MODULATION_INDEX_MAX, step=0.1, 
        value=0.5, description='Modulationsindex:'
    ),
    snr_dB=FloatSlider(
        min=SNR_MIN, max=SNR_MAX, step=1, 
        value=0, description='SNR (dB):'
    ),
    amplitude_signal=FloatSlider(
        min=0.1, max=2.0, step=0.1, 
        value=1.0, description='Baseband-Amplitude (Ω):'
    ),
    amplitude_carrier=FloatSlider(
        min=0.1, max=5.0, step=0.1, 
        value=2.0, description='Träger-Amplitude (Ω):'
    ),
    symbol_rate=IntSlider(
        min=100, max=1e4, step=100, 
        value=5000, description='Symbolrate (Hz):'  # Sichtbar für digitale Typen
    ),
    play_audio_check=play_audio_check
);