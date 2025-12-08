import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def simulate_doppler(f_source, v_observer, v_source, c, duration, sample_rate):
    """
    Simuliert Doppler-Effekt für ein sinusoidales Signal.
    Generiert Original- und beobachtetes Signal, berechnet FFT-Spektren.
    
    Args:
        f_source (float): Originalfrequenz der Quelle (Hz).
        v_observer (float): Beobachtergeschwindigkeit (m/s, + = naht Quelle).
        v_source (float): Quellengeschwindigkeit (m/s, + = entfernt sich vom Beobachter).
        c (float): Übertragungsgeschwindigkeit (z. B. Schall: 343 m/s, LF: Lichtgeschwindigkeit ~3e8 m/s).
        duration (float): Simulationsdauer (s).
        sample_rate (int): Abtastrate (Hz).
    
    Returns:
        dict: Enthält Zeitachse, Original/Doppler-Signale, FFT-Spektren, Frequenzachsen und f_observed.
    """
    # Zeitachse generieren (intervall: 0 ... duration, Punkte: sample_rate * duration)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Doppler-Formel: Beobachtete Frequenz f'
    numerator = c + v_observer
    denominator = c + v_source
    f_observed = f_source * (numerator / denominator)
    
    # Original-Signal (Sinus mit f_source)
    original = np.sin(2 * np.pi * f_source * t)
    # Beobachtetes-Signal (Sinus mit f_observed)
    doppler = np.sin(2 * np.pi * f_observed * t)
    
    # FFT für Original-Signal
    original_fft = fft(original)
    original_fft_mag = np.abs(original_fft) / len(original)  # Amplitude normalisieren
    freqs_original = fftfreq(len(original), 1 / sample_rate)
    # FFT für Doppler-Signal
    doppler_fft = fft(doppler)
    doppler_fft_mag = np.abs(doppler_fft) / len(doppler)
    freqs_doppler = fftfreq(len(doppler), 1 / sample_rate)
    
    # Identifiziere Peaks im FFT-Spektrum (höchste Amplitude)
    peaks_original, _ = find_peaks(original_fft_mag, height=0.1)  # Threshold für Peaks
    peaks_doppler, _ = find_peaks(doppler_fft_mag, height=0.1)
    
    return {
        't': t,
        'original_signal': original,
        'doppler_signal': doppler,
        'f_source': f_source,
        'f_observed': f_observed,
        'sample_rate': sample_rate,
        'original_fft_mag': original_fft_mag,
        'doppler_fft_mag': doppler_fft_mag,
        'freqs_original': freqs_original,
        'freqs_doppler': freqs_doppler,
        'peaks_original': peaks_original,
        'peaks_doppler': peaks_doppler,
        'duration': duration,
        'c': c
    }

def update_plot(sim_data):
    """
    Aktualisiert die Zeit- und Frequenzbereich-Plots mit neuen Simulationsdaten.
    """
    # Zeitbereich-Plot
    ax_time.clear()  # Lösche alte Plot-Daten
    ax_time.plot(sim_data['t'], sim_data['original_signal'], color='#2D708E', linewidth=1.5, 
                 label=f'Original\nf={sim_data["f_source"]:.0f} Hz')
    ax_time.plot(sim_data['t'], sim_data['doppler_signal'], color='#E67E22', linewidth=1.5, 
                 linestyle='--', label=f'Doppler\nf\'={sim_data["f_observed"]:.0f} Hz')
    
    # Markiere Peaks im Zeitbereich (optional, aber hier nicht sichtbar)
    # (Peaks sind im Frequenzbereich relevant)
    
    ax_time.set_title('Zeitbereich: Original vs. Beobachtetes Signal', fontsize=12)
    ax_time.set_xlabel('Zeit (s)', fontsize=10)
    ax_time.set_ylabel('Amplitude', fontsize=10)
    ax_time.grid(True, linestyle='--', alpha=0.6)
    ax_time.legend(fontsize=9, loc='upper right')
    ax_time.set_facecolor('#F8F9FA')  # Hintergrundfarbe für Zeitbereich
    
    # Frequenzbereich-Plot
    ax_freq.clear()
    ax_freq.plot(sim_data['freqs_original'], sim_data['original_fft_mag'], 
                 color='#2D708E', linewidth=1.2, label='Original')
    ax_freq.plot(sim_data['freqs_doppler'], sim_data['doppler_fft_mag'], 
                 color='#E67E22', linewidth=1.2, linestyle='--', label='Doppler')
    
    # Markiere Peaks im FFT-Spektrum
    # Original-Peaks
    if len(sim_data['peaks_original']) > 0:
        peak_freqs_original = sim_data['freqs_original'][sim_data['peaks_original']]  # Frequenzen der Peaks
        peak_mags_original = sim_data['original_fft_mag'][sim_data['peaks_original']]
        ax_freq.scatter(peak_freqs_original, peak_mags_original, 
                       color='#2D708E', marker='x', s=40, zorder=3, label='Original Peaks')
    # Doppler-Peaks
    if len(sim_data['peaks_doppler']) > 0:
        peak_freqs_doppler = sim_data['freqs_doppler'][sim_data['peaks_doppler']]
        peak_mags_doppler = sim_data['doppler_fft_mag'][sim_data['peaks_doppler']]
        ax_freq.scatter(peak_freqs_doppler, peak_mags_doppler, 
                       color='#E67E22', marker='x', s=40, zorder=3, label='Doppler Peaks')
    
    ax_freq.set_title('Frequenzbereich: FFT-Spektren', fontsize=12)
    ax_freq.set_xlabel('Frequenz (Hz)', fontsize=10)
    ax_freq.set_ylabel('Amplitude', fontsize=10)
    ax_freq.grid(True, linestyle='--', alpha=0.6)
    ax_freq.set_xlim(0, sim_data['f_source'] * 3)  # Begrenze Frequenz auf 3×f_source
    ax_freq.legend(fontsize=9, loc='upper right')
    ax_freq.set_facecolor('#F8F9FA')  # Hintergrundfarbe für Frequenzbereich
    
    # Update Plot-Layout
    plt.tight_layout()
    fig.canvas.draw_idle()  # Neue Darstellung anzeigen

def on_slider_change(val):
    """Callback für Sliders: Simuliere neues Signal und aktualisiere Plots."""
    # Hole aktuelle Slider-Werte
    f_source = slider_f_source.val
    v_observer = slider_v_observer.val
    v_source = slider_v_source.val
    duration = slider_duration.val
    sample_rate = int(slider_sample_rate.val)  # Sample-Rate als Integer
    
    # Simuliere Doppler-Effekt mit neuen Parametern
    new_sim_data = simulate_doppler(
        f_source=f_source,
        v_observer=v_observer,
        v_source=v_source,
        c=343,  # Standard-Schallgeschwindigkeit (kann via Slider angepasst werden)
        duration=duration,
        sample_rate=sample_rate
    )
    
    # Update Plots mit neuen Daten
    update_plot(new_sim_data)

def reset_all(event):
    """Callback für Reset-Button: Setze alle Sliders auf Startwerte."""
    # Reset Sliders
    slider_f_source.reset()
    slider_v_observer.reset()
    slider_v_source.reset()
    slider_duration.reset()
    slider_sample_rate.reset()
    # Simuliere mit Startparametern und aktualisiere Plots
    initial_sim_data = simulate_doppler(
        f_source=slider_f_source.valinit,
        v_observer=slider_v_observer.valinit,
        v_source=slider_v_source.valinit,
        c=343,
        duration=slider_duration.valinit,
        sample_rate=int(slider_sample_rate.valinit)
    )
    update_plot(initial_sim_data)

# ---------------------------
# 1. Einstellungen für Widgets
# ---------------------------
# Startparameter (bei Initialisierung der Sliders)
INIT_F_SOURCE = 500    # Originalfrequenz (Hz)
INIT_V_OBSERVER = 10   # Beobachtergeschwindigkeit (m/s)
INIT_V_SOURCE = 0      # Quellengeschwindigkeit (m/s)
INIT_DURATION = 2      # Simulationsdauer (s)
INIT_SAMPLE_RATE = 44100  # Sample-Rate (Hz)

# Slider-Bereiche (min, max)
RANGE_F_SOURCE = (100, 1500)  # Hz
RANGE_V_OBSERVER = (-20, 20)  # m/s (negativ = weg von Quelle)
RANGE_V_SOURCE = (-20, 20)    # m/s (negativ = naht Beobachter)
RANGE_DURATION = (1, 5)       # s
RANGE_SAMPLE_RATE = (1000, 44100)  # Hz (integer)

# ---------------------------
# 2. Plot- und Widget-Setup
# ---------------------------
# Erstelle Figure und Axes für Zeit- und Frequenzbereich
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(16, 10), 
                                      sharex=False, facecolor='white')
fig.suptitle('Doppler-Effekt Simulation (Interaktiv mit Sliders)', fontsize=16, y=0.93)

# Initialisiere Widgets (Sliders) unter den Plots
plt.subplots_adjust(left=0.1, bottom=0.3, right=0.95, top=0.9)  # Reserviere Platz für Widgets

# Slider für Originalfrequenz (f_source)
ax_slider_f = plt.axes([0.1, 0.2, 0.8, 0.03])  # x, y, width, height
slider_f_source = Slider(
    ax=ax_slider_f,
    label='Original Frequenz (Hz)',
    valmin=RANGE_F_SOURCE[0],
    valmax=RANGE_F_SOURCE[1],
    valinit=INIT_F_SOURCE,
    color='#3498DB',  # Blaue Farbe für Slider
    valstep=10  # Schrittgroesse in Hz
)

# Slider für Beobachtergeschwindigkeit (v_observer)
ax_slider_v_obs = plt.axes([0.1, 0.16, 0.8, 0.03])
slider_v_observer = Slider(
    ax=ax_slider_v_obs,
    label='Beobachter Geschwindigkeit (m/s)',
    valmin=RANGE_V_OBSERVER[0],
    valmax=RANGE_V_OBSERVER[1],
    valinit=INIT_V_OBSERVER,
    color='#2ECC71',  # Grün für Beobachter
    valstep=0.5
)

# Slider für Quellengeschwindigkeit (v_source)
ax_slider_v_src = plt.axes([0.1, 0.12, 0.8, 0.03])
slider_v_source = Slider(
    ax=ax_slider_v_src,
    label='Quelle Geschwindigkeit (m/s)',
    valmin=RANGE_V_SOURCE[0],
    valmax=RANGE_V_SOURCE[1],
    valinit=INIT_V_SOURCE,
    color='#E74C3C',  # Rot für Quelle
    valstep=0.5
)

# Slider für Simulationsdauer (duration)
ax_slider_dur = plt.axes([0.1, 0.08, 0.8, 0.03])
slider_duration = Slider(
    ax=ax_slider_dur,
    label='Dauer (s)',
    valmin=RANGE_DURATION[0],
    valmax=RANGE_DURATION[1],
    valinit=INIT_DURATION,
    color='#F1C40F',  # Gelb für Dauer
    valstep=0.5
)

# Slider für Sample-Rate (sample_rate)
ax_slider_sr = plt.axes([0.1, 0.04, 0.8, 0.03])
slider_sample_rate = Slider(
    ax=ax_slider_sr,
    label='Sample-Rate (Hz)',
    valmin=RANGE_SAMPLE_RATE[0],
    valmax=RANGE_SAMPLE_RATE[1],
    valinit=INIT.Sample_RATE,
    color='#9B59B6',  # Lila für Sample-Rate
    valstep=100
)

# Reset-Button
ax_button_reset = plt.axes([0.7, 0.01, 0.2, 0.05])
reset_button = Button(
    ax=ax_button_reset,
    label='Parameter zurücksetzen',
    color='lightgray',
    hovercolor='#F5B7B1'
)
reset_button.on_clicked(reset_all)

# ---------------------------
# 3. Initialisiere Simulationsdaten und Plot
# ---------------------------
# Erstelle Initialdaten
initial_data = simulate_doppler(
    f_source=INIT_F_SOURCE,
    v_observer=INIT_V_OBSERVER,
    v_source=INIT_V_SOURCE,
    c=343,
    duration=INIT_DURATION,
    sample_rate=INIT_SAMPLE_RATE
)
# Zeichne Initialplots
update_plot(initial_data)

# ---------------------------
# 4. Verbinde Sliders mit Callback-Funktion
# ---------------------------
slider_f_source.on_changed(on_slider_change)
slider_v_observer.on_changed(on_slider_change)
slider_v_source.on_changed(on_slider_change)
slider_duration.on_changed(on_slider_change)
slider_sample_rate.on_changed(on_slider_change)

plt.show()