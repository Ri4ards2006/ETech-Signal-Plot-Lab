import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
        c (float): Übertragungsgeschwindigkeit (z. B. Schall: 343 m/s, Licht: 3e8 m/s).
        duration (float): Simulationsdauer (s).
        sample_rate (int): Abtastrate (Hz).
    
    Returns:
        dict: Enthält Zeitachse, Signale, FFT-Daten und berechnete Werte.
    """
    # Zeitachse generieren (Punkte: sample_rate * duration)
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Doppler-Formel: Beobachtete Frequenz f'
    denominator = c + v_source if c + v_source != 0 else 1e-9  # Vermeide Division durch 0
    f_observed = f_source * ( (c + v_observer) / denominator )
    
    # Original- und Doppler-Signal (Amplitude=1)
    original = np.sin(2 * np.pi * f_source * t)
    doppler = np.sin(2 * np.pi * f_observed * t)
    
    # FFT optimieren (nur positive Frequenzen)
    def compute_fft(signal, sample_rate):
        fft_vals = fft(signal)
        fft_mag = np.abs(fft_vals) / len(signal)  # Normalisiere Amplitude
        freqs = fftfreq(len(signal), 1 / sample_rate)
        pos_mask = freqs > 0  # Nur positive Frequenzen
        return freqs[pos_mask], fft_mag[pos_mask]
    
    freqs_original, original_fft_mag = compute_fft(original, sample_rate)
    freqs_doppler, doppler_fft_mag = compute_fft(doppler, sample_rate)
    
    # Peaks finden (höchste Amplituden)
    peaks_original, _ = find_peaks(original_fft_mag, height=0.02)  # Niedrigerer Threshold
    peaks_doppler, _ = find_peaks(doppler_fft_mag, height=0.02)
    
    return {
        't': t,
        'original_signal': original,
        'doppler_signal': doppler,
        'f_source': f_source,
        'f_observed': f_observed,
        'c': c,
        'duration': duration,
        'sample_rate': sample_rate,
        'freqs_original': freqs_original,
        'original_fft_mag': original_fft_mag,
        'freqs_doppler': freqs_doppler,
        'doppler_fft_mag': doppler_fft_mag,
        'peaks_original': peaks_original,
        'peaks_doppler': peaks_doppler,
        'v_observer': v_observer,
        'v_source': v_source
    }

def update_plot(sim_data):
    """
    Aktualisiert die Zeit- und Frequenzbereich-Plots mit neuen Simulationsdaten.
    """
    # Zeitbereich-Plot (oben)
    ax_time.clear()
    ax_time.plot(sim_data['t'], sim_data['original_signal'], color='#2D708E', linewidth=1.5, label='Original')
    ax_time.plot(sim_data['t'], sim_data['doppler_signal'], color='#E67E22', linewidth=1.5, linestyle='--', label='Doppler')
    
    # Aktuelle Parameter im Zeitbereich anzeigen
    ax_time.text(0.05, 0.9, 
                 f'f_source: {sim_data["f_source"]:.0f} Hz\n'
                 f'v_observer: {sim_data["v_observer"]:.1f} m/s\n'
                 f'v_source: {sim_data["v_source"]:.1f} m/s\n'
                 f'Sample-Rate: {sim_data["sample_rate"]:.0f} Hz',
                 transform=ax_time.transAxes, 
                 fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.9)
                )
    
    ax_time.set_title('Zeitbereich: Original vs. Doppler-Signal', fontsize=12, pad=15)
    ax_time.set_xlabel('Zeit (s)')
    ax_time.set_ylabel('Amplitude')
    ax_time.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC')
    ax_time.legend(loc='upper right', fontsize=9)
    ax_time.set_facecolor('#F8F9FA')  # Weißlicher Hintergrund
    ax_time.set_xlim(0, sim_data['duration'])  # Zeitachse anpassen

    # Frequenzbereich-Plot (unten)
    ax_freq.clear()
    ax_freq.plot(sim_data['freqs_original'], sim_data['original_fft_mag'], 
                 color='#2D708E', linewidth=1.2, label=f'Original (f={sim_data["f_source"]:.0f} Hz)')
    ax_freq.plot(sim_data['freqs_doppler'], sim_data['doppler_fft_mag'], 
                 color='#E67E22', linewidth=1.2, linestyle='--', label=f'Doppler (f\'={sim_data["f_observed"]:.0f} Hz)')
    
    # Peaks markieren
    if len(sim_data['peaks_original']) > 0:
        peak_freqs_orig = sim_data['freqs_original'][sim_data['peaks_original']]
        ax_freq.scatter(peak_freqs_orig, sim_data['original_fft_mag'][sim_data['peaks_original']], 
                       color='#2D708E', marker='x', s=70, zorder=3, label='Original Peak')
    if len(sim_data['peaks_doppler']) > 0:
        peak_freqs_dop = sim_data['freqs_doppler'][sim_data['peaks_doppler']]
        ax_freq.scatter(peak_freqs_dop, sim_data['doppler_fft_mag'][sim_data['peaks_doppler']], 
                       color='#E67E22', marker='x', s=70, zorder=3, label='Doppler Peak')
    
    # Frequenzbereich anpassen
    ax_freq.set_xlim(0, sim_data['f_source'] * 2)
    ax_freq.set_ylim(0, max(sim_data['original_fft_mag'].max(), sim_data['doppler_fft_mag'].max()) * 1.1)
    
    ax_freq.set_title('Frequenzbereich: FFT-Spektren', fontsize=12, pad=15)
    ax_freq.set_xlabel('Frequenz (Hz)')
    ax_freq.set_ylabel('Amplitude')
    ax_freq.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC')
    ax_freq.legend(loc='upper right', fontsize=9)
    ax_freq.set_facecolor('#F8F9FA')  # Gleiches Hintergrunddesign

    plt.tight_layout()
    fig.canvas.draw()  # Schneller Refresh

def on_slider_change(val):
    """Callback: Simuliere neues Signal und aktualisiere Plot."""
    # Slider-Werte abrufen
    f_source = slider_f_source.val
    v_observer = slider_v_observer.val
    v_source = slider_v_source.val
    duration = slider_duration.val
    sample_rate = int(slider_sample_rate.val)
    c = 343  # Schallgeschwindigkeit (kann via Slider angepasst werden)
    
    # Neues Signal simulieren und Plot aktualisieren
    sim_data = simulate_doppler(f_source, v_observer, v_source, c, duration, sample_rate)
    update_plot(sim_data)

def reset_all(event):
    """Callback: Reset Sliders und Plot."""
    # Sliders auf Startwerte zurücksetzen
    slider_f_source.reset()
    slider_v_observer.reset()
    slider_v_source.reset()
    slider_duration.reset()
    slider_sample_rate.reset()
    # Startdaten simulieren und Plot aktualisieren
    initial_data = simulate_doppler(
        INIT_F_SOURCE, INIT_V_OBSERVER, INIT_V_SOURCE, 
        INIT_C, INIT_DURATION, INIT_SAMPLE_RATE
    )
    update_plot(initial_data)

# ---------------------------
# 1. Startparameter & Slider-Bereiche
# ---------------------------
INIT_F_SOURCE = 500       # Originalfrequenz (Hz)
INIT_V_OBSERVER = 10      # Beobachter Geschwindigkeit (m/s)
INIT_V_SOURCE = 0         # Quelle Geschwindigkeit (m/s)
INIT_DURATION = 2         # Simulationsdauer (s)
INIT_SAMPLE_RATE = 44100  # Sample-Rate (Hz)
INIT_C = 343              # Übertragungsgeschwindigkeit (m/s, Schall)

RANGE_F_SOURCE = (100, 1500)      # Hz
RANGE_V_OBSERVER = (-20, 20)      # m/s
RANGE_V_SOURCE = (-20, 20)        # m/s
RANGE_DURATION = (1, 5)            # s
RANGE_SAMPLE_RATE = (1000, 44100)  # Hz

# ---------------------------
# 2. Plot-Setup (Figure & Axes)
# ---------------------------
fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(14, 8), facecolor='#FFFFFF')
fig.suptitle('Interaktive Doppler-Effekt-Simulation', fontsize=16, y=0.95)
plt.subplots_adjust(left=0.1, bottom=0.4, right=0.95, top=0.9)  # Platz für Widgets

# ---------------------------
# 3. Slider-Erstellung (Widgets) ▷ FEHLER BEHEBt!
# ---------------------------
# Slider für Originalfrequenz (f_source)
ax_slider_f = plt.axes([0.1, 0.3, 0.8, 0.03])  # x, y, width, height
slider_f_source = Slider(
    ax=ax_slider_f,
    label='Original Frequenz (Hz)',
    valmin=RANGE_F_SOURCE[0],
    valmax=RANGE_F_SOURCE[1],
    valinit=INIT_F_SOURCE,
    valstep=10,  # Schrittgröße
    color='#3498DB',  # Farbe des Knopfs (statt 'valcolor')
    track_color='#E3F2FD'  # Farbe des Tracks (statt 'facecolor')
)

# Slider für Beobachtergeschwindigkeit (v_observer)
ax_slider_v_obs = plt.axes([0.1, 0.25, 0.8, 0.03])
slider_v_observer = Slider(
    ax=ax_slider_v_obs,
    label='Beobachter Geschwindigkeit (m/s)',
    valmin=RANGE_V_OBSERVER[0],
    valmax=RANGE_V_OBSERVER[1],
    valinit=INIT_V_OBSERVER,
    valstep=0.5,
    color='#2ECC71',  # Grün für Beobachter-Knopf
    track_color='#F0FFF0'  # Hellgrün für Track
)

# Slider für Quellengeschwindigkeit (v_source)
ax_slider_v_src = plt.axes([0.1, 0.2, 0.8, 0.03])
slider_v_source = Slider(
    ax=ax_slider_v_src,
    label='Quelle Geschwindigkeit (m/s)',
    valmin=RANGE_V_SOURCE[0],
    valmax=RANGE_V_SOURCE[1],
    valinit=INIT_V_SOURCE,
    valstep=0.5,
    color='#E74C3C',  # Rot für Quelle-Knopf
    track_color='#FFF0F5'  # Hellrot für Track
)

# Slider für Simulationsdauer (duration)
ax_slider_dur = plt.axes([0.1, 0.15, 0.8, 0.03])
slider_duration = Slider(
    ax=ax_slider_dur,
    label='Simulationsdauer (s)',
    valmin=RANGE_DURATION[0],
    valmax=RANGE_DURATION[1],
    valinit=INIT_DURATION,
    valstep=0.5,
    color='#F1C40F',  # Gelb für Dauer-Knopf
    track_color='#FFF8E7'  # Hellgelb für Track
)

# Slider für Sample-Rate (sample_rate)
ax_slider_sr = plt.axes([0.1, 0.1, 0.8, 0.03])
slider_sample_rate = Slider(
    ax=ax_slider_sr,
    label='Sample-Rate (Hz)',
    valmin=RANGE_SAMPLE_RATE[0],
    valmax=RANGE_SAMPLE_RATE[1],
    valinit=INIT_SAMPLE_RATE,
    valstep=100,
    color='#9B59B6',  # Lila für Sample-Rate-Knopf
    track_color='#E6E6FA'  # Lavendelfarben für Track
)

# ---------------------------
# 4. Reset-Button (ohne 'fontcolor'-Argument)
# ---------------------------
ax_button_reset = plt.axes([0.7, 0.03, 0.2, 0.05])
reset_button = Button(
    ax=ax_button_reset,
    label='Parameter zurücksetzen',
    color='#ECF0F1',  # Normal-Background (Hellgrau)
    hovercolor='#F5B7B1'  # Hover-Background (Orange-Grau)
)
# Schriftfarbe manuell setzen
reset_button.label.set_color('#2C3E50')  # Dunkler Text

reset_button.on_clicked(reset_all)

# ---------------------------
# 5. Initialisierung und Start
# ---------------------------
initial_data = simulate_doppler(
    INIT_F_SOURCE, INIT_V_OBSERVER, INIT_V_SOURCE, 
    INIT_C, INIT_DURATION, INIT_SAMPLE_RATE
)
update_plot(initial_data)

# Sliders an Callback-Funktion binden
slider_f_source.on_changed(on_slider_change)
slider_v_observer.on_changed(on_slider_change)
slider_v_source.on_changed(on_slider_change)
slider_duration.on_changed(on_slider_change)
slider_sample_rate.on_changed(on_slider_change)

plt.show()