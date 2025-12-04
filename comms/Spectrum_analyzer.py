import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.backends.backend_tkagg as tkagg

class SpectrumAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrum Analyzer Emulator")
        
        # Signalvariablen
        self.signal = None
        self.Fs = None  # Abtastrate
        self.t = None   # Zeitachse
        self.freqs_positive = None  # Positive Frequenzen (FFT)
        self.db_magnitude_positive = None  # dB-Amplitude (FFT)
        self.RBW = None  # Resolution Bandwidth
        self.markers = []  # Markierungen (Frequenz, dB)
        
        # Statusvariablen
        self.status_var = tk.StringVar()
        self.current_signal_type = 'Sinus'

        # GUI-Elemente erstellen
        self.create_widgets()
        self.update_parameters()

    def create_widgets(self):
        # Parameter Frame
        self.params_frame = ttk.Frame(self.root)
        self.params_frame.pack(padx=10, pady=10, fill='x')

        # Signal-Typ Auswahl
        self.signal_type_label = ttk.Label(self.params_frame, text="Signal Typ:")
        self.signal_type_label.pack(side='left')
        
        self.signal_type = ttk.Combobox(self.params_frame, 
                                      values=['Sinus', 'FM-Ton', 'LTE-Frame', 'WAV-File'], 
                                      state='readonly')
        self.signal_type.pack(side='left', padx=5)
        self.signal_type.bind('<<ComboboxSelected>>', self.update_parameters)

        # Dynamischer Parameterbereich
        self.dynamic_params_frame = ttk.Frame(self.params_frame)
        self.dynamic_params_frame.pack(side='left', padx=10, fill='x', expand=True)

        # Buttons Frame
        self.buttons_frame = ttk.Frame(self.root)
        self.buttons_frame.pack(padx=10, pady=5, fill='x')

        self.generate_btn = ttk.Button(self.buttons_frame, 
                                      text="Generate/Load Signal", 
                                      command=self.generate_or_load_signal)
        self.generate_btn.pack(side='left', padx=5)

        self.plot_btn = ttk.Button(self.buttons_frame, 
                                  text="Plot Spectrum", 
                                  command=self.plot_spectrum)
        self.plot_btn.pack(side='left', padx=5)

        # Zoom-Einstellungen
        self.zoom_frame = ttk.Frame(self.buttons_frame)
        self.zoom_frame.pack(side='left', padx=10)

        ttk.Label(self.zoom_frame, text="Zoom (Hz):").grid(row=0, column=0, padx=5)
        self.zoom_start = ttk.Entry(self.zoom_frame, width=8)
        self.zoom_start.grid(row=0, column=1, padx=5)
        self.zoom_start.insert(0, '0')

        self.zoom_end = ttk.Entry(self.zoom_frame, width=8)
        self.zoom_end.grid(row=0, column=2, padx=5)
        self.zoom_end.insert(0, '22050')

        self.zoom_btn = ttk.Button(self.zoom_frame, text="Zoom", command=self.zoom_spectrum)
        self.zoom_btn.grid(row=0, column=3, padx=5)

        # Messbereich-Einstellungen
        self.measure_frame = ttk.Frame(self.root)
        self.measure_frame.pack(padx=10, pady=5, fill='x')

        ttk.Label(self.measure_frame, text="Messbereich (Hz):").grid(row=0, column=0, padx=5)
        self.measure_start = ttk.Entry(self.measure_frame, width=8)
        self.measure_start.grid(row=0, column=1, padx=5)
        self.measure_start.insert(0, '0')

        self.measure_end = ttk.Entry(self.measure_frame, width=8)
        self.measure_end.grid(row=0, column=2, padx=5)
        self.measure_end.insert(0, '22050')

        self.measure_btn = ttk.Button(self.measure_frame, 
                                     text="Measure", 
                                     command=self.measure_signal)
        self.measure_btn.grid(row=0, column=3, padx=5)

        self.peak_label = ttk.Label(self.measure_frame, text="Peak dB: ")
        self.peak_label.grid(row=0, column=4, padx=5)

        self.rms_label = ttk.Label(self.measure_frame, text="RMS dB: ")
        self.rms_label.grid(row=0, column=5, padx=5)

        # Statusanzeige
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(padx=10, pady=5, fill='x')
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side='left')
        self.status_var.set("Bereit.")

        # Plot-Bereich
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ax = self.fig.add_subplot(111)

        # Klick-Ereignis für Markierungen
        self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)

    def update_parameters(self, event=None):
        # Alt Widget löschen
        for widget in self.dynamic_params_frame.winfo_children():
            widget.destroy()
        
        self.current_signal_type = self.signal_type.get()
        # Neue Parameterfelder anlegen
        if self.current_signal_type == 'Sinus':
            self.create_sinus_params()
        elif self.current_signal_type == 'FM-Ton':
            self.create_fm_params()
        elif self.current_signal_type == 'LTE-Frame':
            self.create_lte_params()
        elif self.current_signal_type == 'WAV-File':
            self.create_wav_params()
        else:
            self.status_var.set("Unbekannter Signaltyp")

    def create_sinus_params(self):
        # Sinus-Parameter
        ttk.Label(self.dynamic_params_frame, text="Amplitude:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.sin_amp = ttk.Entry(self.dynamic_params_frame)
        self.sin_amp.grid(row=0, column=1, padx=5, pady=5)
        self.sin_amp.insert(0, '1.0')

        ttk.Label(self.dynamic_params_frame, text="Frequenz (Hz):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.sin_freq = ttk.Entry(self.dynamic_params_frame)
        self.sin_freq.grid(row=1, column=1, padx=5, pady=5)
        self.sin_freq.insert(0, '1000')

        ttk.Label(self.dynamic_params_frame, text="Dauer (s):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.sin_duration = ttk.Entry(self.dynamic_params_frame)
        self.sin_duration.grid(row=2, column=1, padx=5, pady=5)
        self.sin_duration.insert(0, '1.0')

        ttk.Label(self.dynamic_params_frame, text="Abtastrate (Hz):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.sin_fs = ttk.Entry(self.dynamic_params_frame)
        self.sin_fs.grid(row=3, column=1, padx=5, pady=5)
        self.sin_fs.insert(0, '44100')

    def create_fm_params(self):
        # FM-Parameter
        ttk.Label(self.dynamic_params_frame, text="Amplitude:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.fm_amp = ttk.Entry(self.dynamic_params_frame)
        self.fm_amp.grid(row=0, column=1, padx=5, pady=5)
        self.fm_amp.insert(0, '1.0')

        ttk.Label(self.dynamic_params_frame, text="Trägerfrequenz (Hz):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.fm_fc = ttk.Entry(self.dynamic_params_frame)
        self.fm_fc.grid(row=1, column=1, padx=5, pady=5)
        self.fm_fc.insert(0, '10000')

        ttk.Label(self.dynamic_params_frame, text="Modulationsfreq (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.fm_fm = ttk.Entry(self.dynamic_params_frame)
        self.fm_fm.grid(row=2, column=1, padx=5, pady=5)
        self.fm_fm.insert(0, '1000')

        ttk.Label(self.dynamic_params_frame, text="Modulationsindex (β):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.fm_beta = ttk.Entry(self.dynamic_params_frame)
        self.fm_beta.grid(row=3, column=1, padx=5, pady=5)
        self.fm_beta.insert(0, '1.0')

        ttk.Label(self.dynamic_params_frame, text="Dauer (s):").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.fm_duration = ttk.Entry(self.dynamic_params_frame)
        self.fm_duration.grid(row=4, column=1, padx=5, pady=5)
        self.fm_duration.insert(0, '1.0')

        ttk.Label(self.dynamic_params_frame, text="Abtastrate (Hz):").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.fm_fs = ttk.Entry(self.dynamic_params_frame)
        self.fm_fs.grid(row=5, column=1, padx=5, pady=5)
        self.fm_fs.insert(0, '44100')

    def create_lte_params(self):
        # LTE-Parameter
        ttk.Label(self.dynamic_params_frame, text="Amplitude:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.lte_amp = ttk.Entry(self.dynamic_params_frame)
        self.lte_amp.grid(row=0, column=1, padx=5, pady=5)
        self.lte_amp.insert(0, '1.0')

        ttk.Label(self.dynamic_params_frame, text="Trägerfrequenz (Hz):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.lte_fc = ttk.Entry(self.dynamic_params_frame)
        self.lte_fc.grid(row=1, column=1, padx=5, pady=5)
        self.lte_fc.insert(0, '700000000')

        ttk.Label(self.dynamic_params_frame, text="Bandbreite (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.lte_bw = ttk.Entry(self.dynamic_params_frame)
        self.lte_bw.grid(row=2, column=1, padx=5, pady=5)
        self.lte_bw.insert(0, '10000000')

        ttk.Label(self.dynamic_params_frame, text="Dauer (s):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.lte_duration = ttk.Entry(self.dynamic_params_frame)
        self.lte_duration.grid(row=3, column=1, padx=5, pady=5)
        self.lte_duration.insert(0, '0.001')

        ttk.Label(self.dynamic_params_frame, text="Abtastrate (Hz):").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.lte_fs = ttk.Entry(self.dynamic_params_frame)
        self.lte_fs.grid(row=4, column=1, padx=5, pady=5)
        self.lte_fs.insert(0, '20000000')

    def create_wav_params(self):
        # WAV-Parameter
        ttk.Label(self.dynamic_params_frame, text="Dateipfad:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.wav_path = ttk.Entry(self.dynamic_params_frame)
        self.wav_path.grid(row=0, column=1, padx=5, pady=5)

        self.wav_browse = ttk.Button(self.dynamic_params_frame, 
                                    text="Browse", 
                                    command=self.browse_wav)
        self.wav_browse.grid(row=0, column=2, padx=5, pady=5)

    def browse_wav(self):
        path = filedialog.askopenfilename(filetypes=[("WAV Dateien", "*.wav")])
        if path:
            self.wav_path.delete(0, tk.END)
            self.wav_path.insert(0, path)

    def generate_or_load_signal(self):
        self.markers = []
        self.ax.clear()
        self.canvas.draw()
        
        try:
            if self.current_signal_type == 'Sinus':
                self.generate_sinus()
            elif self.current_signal_type == 'FM-Ton':
                self.generate_fm()
            elif self.current_signal_type == 'LTE-Frame':
                self.generate_lte()
            elif self.current_signal_type == 'WAV-File':
                self.load_wav()
            else:
                self.status_var.set("Signaltyp nicht unterstützt")
                return

            self.status_var.set("Signal erfolgreich geladen/erzeugt")
            # Zoom-Einstellungen aktualisieren
            if self.Fs:
                self.zoom_end.delete(0, tk.END)
                self.zoom_end.insert(0, f'{self.Fs/2:.0f}')
                self.measure_end.delete(0, tk.END)
                self.measure_end.insert(0, f'{self.Fs/2:.0f}')
                
        except Exception as e:
            self.status_var.set(f"Fehler: {str(e)}")

    def generate_sinus(self):
        A = float(self.sin_amp.get())
        f = float(self.sin_freq.get())
        T = float(self.sin_duration.get())
        Fs = float(self.sin_fs.get())
        
        N = int(Fs * T)
        self.t = np.linspace(0, T, N, endpoint=False)
        self.signal = A * np.sin(2 * np.pi * f * self.t)
        self.Fs = Fs

    def generate_fm(self):
        A = float(self.fm_amp.get())
        fc = float(self.fm_fc.get())
        fm = float(self.fm_fm.get())
        beta = float(self.fm_beta.get())
        T = float(self.fm_duration.get())
        Fs = float(self.fm_fs.get())
        
        N = int(Fs * T)
        self.t = np.linspace(0, T, N, endpoint=False)
        mod_sig = np.sin(2 * np.pi * fm * self.t)  # Modulationsignal
        phase = 2 * np.pi * fc * self.t + beta * mod_sig
        self.signal = A * np.cos(phase)
        self.Fs = Fs

    def generate_lte(self):
        A = float(self.lte_amp.get())
        fc = float(self.lte_fc.get())
        BW = float(self.lte_bw.get())
        T = float(self.lte_duration.get())
        Fs = float(self.lte_fs.get())
        
        # Prüfung Abtastrate
        if Fs <= 2 * BW:
            raise ValueError(f"Abtastrate {Fs}Hz zu niedrig (min {2*BW}Hz für Basisband)")
        
        N = int(Fs * T)
        self.t = np.linspace(0, T, N, endpoint=False)
        
        # Basisband-Signal (Gaußscher Rauschen gefiltert)
        noise = np.random.normal(0, 1, N)
        nyq = 0.5 * Fs
        cutoff = BW / 2
        b, a = signal.butter(5, cutoff/nyq, 'low')
        baseband = signal.lfilter(b, a, noise)
        
        # Skalierung auf Amplitude A
        peak = np.max(np.abs(baseband))
        if peak == 0:
            baseband_scaled = baseband
        else:
            baseband_scaled = baseband * (A / peak)
        
        # Auftragen auf Trägerfrequenz
        self.signal = baseband_scaled * np.cos(2 * np.pi * fc * self.t)
        self.Fs = Fs

    def load_wav(self):
        path = self.wav_path.get().strip()
        if not path:
            raise ValueError("Kein Dateipfad angegeben")
        
        Fs, data = signal.wavfile.read(path)
        if len(data.shape) > 1:
            data = data[:, 0]  # Mono-Kanal
        
        # Konvertierung in Float
        data = data.astype(np.float64) / (2**15 - 1)  # Für 16-Bit
        self.signal = data
        self.Fs = Fs
        self.t = np.linspace(0, len(data)/Fs, len(data), endpoint=False)

    def process_fft(self):
        if self.signal is None or self.Fs is None:
            return
        
        # Fensterung (Hanning) für bessere Spektraleausexy
        window = np.hanning(len(self.signal))
        windowed = self.signal * window
        
        # FFT berechnen
        fft_vals = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(len(fft_vals), 1/self.Fs)
        freqs_shifted = np.fft.fftshift(freqs)
        fft_shifted = np.fft.fftshift(fft_vals)
        
        # dB-Umrechnung (relativ zum Peak)
        magnitude = np.abs(fft_shifted)
        if np.max(magnitude) == 0:
            db_mag = np.zeros_like(magnitude)
        else:
            db_mag = 20 * np.log10(magnitude / np.max(magnitude))
        
        # Nur positive Frequenzen betrachten
        pos_mask = freqs_shifted >= 0
        self.freqs_positive = freqs_shifted[pos_mask]
        self.db_magnitude_positive = db_mag[pos_mask]
        
        # Resolution Bandwidth (RBW) berechnen
        self.RBW = self.Fs / len(self.signal)

    def plot_spectrum(self):
        self.process_fft()
        if self.freqs_positive is None or self.db_magnitude_positive is None:
            self.status_var.set("Kein Signal zum Plotten")
            return

        self.ax.clear()
        self.ax.plot(self.freqs_positive, self.db_magnitude_positive)
        self.ax.set_xlabel('Frequenz (Hz)')
        self.ax.set_ylabel('Amplitude (dB)')
        self.ax.set_title('Spektrumanalyse')
        self.ax.set_xlim(0, self.Fs/2)
        self.ax.autoscale(axis='y', tight=True)

        # Markierungen einblenden
        for freq in self.markers:
            idx = np.argmin(np.abs(self.freqs_positive - freq))
            db = self.db_magnitude_positive[idx]
            self.ax.axvline(x=freq, color='r', linestyle='--')
            self.ax.text(freq, self.ax.get_ylim()[1], 
                        f'{freq:.2f}Hz\n{db:.2f}dB', 
                        rotation=90, va='top', ha='center', 
                        color='r', bbox=dict(facecolor='white', alpha=0.8))

        self.canvas.draw()

    def on_plot_click(self, event):
        if event.inaxes != self.ax:
            return
        
        freq = event.xdata
        if self.freqs_positive is None or self.db_magnitude_positive is None:
            return
        
        # Näste Frequenz finden
        idx = np.argmin(np.abs(self.freqs_positive - freq))
        db = self.db_magnitude_positive[idx]
        self.markers.append(freq)  # Markierung an Frequenz speichern
        self.plot_spectrum()  # Plot aktualisieren

    def zoom_spectrum(self):
        try:
            f_start = float(self.zoom_start.get())
            f_end = float(self.zoom_end.get())
            
            if f_start >= f_end:
                raise ValueError("Startfrequenz >= Endfrequenz")
            
            max_freq = self.Fs / 2 if self.Fs else 0
            f_start = max(f_start, 0)
            f_end = min(f_end, max_freq)
            
            self.ax.set_xlim(f_start, f_end)
            self.canvas.draw()
        except ValueError as e:
            self.status_var.set(f"Zoom Fehler: {e}")

    def measure_signal(self):
        if self.signal is None or self.Fs is None:
            self.status_var.set("Kein Signal zum Messen")
            self.peak_label.config(text="Peak dB: -")
            self.rms_label.config(text="RMS dB: -")
            return

        try:
            f_start = float(self.measure_start.get())
            f_end = float(self.measure_end.get())

            if f_start >= f_end:
                raise ValueError("Startfrequenz >= Endfrequenz")

            # Bandpass-Filter design
            nyq = 0.5 * self.Fs
            low = f_start / nyq
            high = f_end / nyq
            b, a = signal.butter(5, [low, high], 'band')

            # Filter anwenden
            filtered = signal.lfilter(b, a, self.signal)

            # Peak und RMS berechnen
            peak = np.max(np.abs(filtered))
            rms = np.sqrt(np.mean(filtered**2))

            # dB-Umrechnung
            peak_db = 20 * np.log10(peak) if peak > 1e-10 else -np.inf
            rms_db = 20 * np.log10(rms) if rms > 1e-10 else -np.inf

            self.peak_label.config(text=f"Peak dB: {peak_db:.2f}")
            self.rms_label.config(text=f"RMS dB: {rms_db:.2f}")
            self.status_var.set("Messung erfolgreich")
        except ValueError as e:
            self.status_var.set(f"Messung Fehler: {e}")
            self.peak_label.config(text="Peak dB: -")
            self.rms_label.config(text="RMS dB: -")

def main():
    root = tk.Tk()
    app = SpectrumAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()