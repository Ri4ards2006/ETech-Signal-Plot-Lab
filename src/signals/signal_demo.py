import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
fs = 1000           # Sampling frequency
t = np.arange(0, 1, 1/fs)  # 1 second of data

# --- Signals ---
sin_wave = 2 * np.sin(2 * np.pi * 5 * t)              # 5 Hz sine wave
square_wave = np.sign(np.sin(2 * np.pi * 10 * t))    # 10 Hz square wave
noise = 0.5 * np.random.randn(len(t))                # white noise

# --- Combined signal ---
combined = sin_wave + square_wave + noise

# --- FFT ---
fft_vals = np.fft.fft(combined)
fft_freq = np.fft.fftfreq(len(t), 1/fs)

# --- Plot time-domain signals ---
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, combined, label='Combined Signal 🌊')
plt.plot(t, sin_wave, '--', label='Sine Wave')
plt.plot(t, square_wave, '--', label='Square Wave')
plt.title("Time Domain Signals")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# --- Plot frequency-domain (FFT) ---
plt.subplot(2, 1, 2)
plt.plot(fft_freq[:len(t)//2], np.abs(fft_vals)[:len(t)//2], color='orange')
plt.title("Frequency Spectrum 🔊")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
