# ============================================================================
# FFT Demonstration Script for ETech-Signal-Plot-Lab
# Purpose: Analyze the frequency components of a signal using the Fast Fourier Transform (FFT)
# Author: [Your Name] | Date: [Insert Date]
# ============================================================================

# ------------------------------
# 1. Import Required Libraries
# ------------------------------
# NumPy: Fundamental package for numerical computations (signal generation, FFT)
import numpy as np
# Matplotlib: Plotting library to visualize time-domain and frequency-domain signals
import matplotlib.pyplot as plt

# ------------------------------
# 2. Set Core Parameters
# ------------------------------
# Sampling Rate (fs): Number of samples collected per second (Hz)
# CRITICAL: Must be at least twice the highest frequency in the signal (Nyquist-Shannon Theorem)
fs = 1000  # Hz (1 kHz sampling rate)
print(f"Sampling Rate (fs): {fs} Hz")  # Log parameters for transparency

# Total Observation Time (T): Duration for which the signal is measured (seconds)
T = 1.0  # 1 second of observation
print(f"Total Observation Time (T): {T}s")

# Number of Samples (N): Total points in the time-domain signal (fs * T)
# Explicitly defined here for clarity; alternatively, N = len(t) later
N = int(fs * T)  # fs*T = 1000*1 = 1000 → 1000 samples
print(f"Number of Samples (N): {N}")

# ------------------------------
# 3. Generate Time Array (Time Axis)
# ------------------------------
# Time array 't' contains evenly spaced time values from 0 to T (excluding T)
# Use endpoint=False to avoid overlapping the last sample with the start of the next period (for periodic signals)
t = np.linspace(0, T, N, endpoint=False)  # Time axis: [0, 1), 1000 points
print(f"Length of time array (t): {len(t)} → Matches N: {len(t) == N}")  # Verify consistency

# ------------------------------
# 4. Generate Signal in Time Domain
# ------------------------------
# Let's create a test signal: 50 Hz sine wave + 150 Hz sine wave + white noise
# This demonstrates how FFT extracts multiple frequency components

# Define signal frequencies and amplitudes
freq1 = 50    # Hz (primary frequency component)
freq2 = 150   # Hz (secondary frequency component)
amplitude1 = 2.0  # Amplitude of freq1 (larger for visibility)
amplitude2 = 1.0  # Amplitude of freq2

# Generate sine wave for freq1: A * sin(2πft)
signal_freq1 = amplitude1 * np.sin(2 * np.pi * freq1 * t)
# Generate sine wave for freq2
signal_freq2 = amplitude2 * np.sin(2 * np.pi * freq2 * t)
# Generate white noise: Random values with mean 0 (simulates real-world noise)
noise_amplitude = 0.5  # Controls noise strength
noise = noise_amplitude * np.random.normal(0, 1, N)  # Normal distribution (mean=0, std=1)
# Combine all components into the total signal
signal = signal_freq1 + signal_freq2 + noise

# ------------------------------
# 5. Plot the Time-Domain Signal
# ------------------------------
# Visualize the original signal to understand its time-domain behavior
plt.figure(figsize=(12, 6))  # Set figure size (width, height) in inches
plt.subplot(2, 1, 1)  # Create 2 rows, 1 column; first plot (top)

# Plot time vs. signal amplitude
plt.plot(t, signal, label='Total Signal (Time Domain)')
plt.xlabel('Time (s)')  # Label x-axis
plt.ylabel('Amplitude')  # Label y-axis
plt.title('Original Signal in Time Domain')  # Plot title
plt.xlim(0, T)  # Limit x-axis to 0–1s (full observation time)
plt.ylim(-4, 4)  # Limit y-axis to accommodate max possible amplitude (signal_freq1 + signal_freq2 + noise)
plt.legend()  # Show legend with labels
plt.grid(True, linestyle='--')  # Add grid lines for readability

# ------------------------------
# 6. Compute FFT (Frequency Domain Conversion)
# ------------------------------
# FFT (Fast Fourier Transform): Converts a time-domain signal to its frequency-domain representation
# Input: Time-domain signal (real or complex); Output: Complex array (amplitude + phase information)

# Step 6.1: Compute FFT of the signal
fft_values = np.fft.fft(signal)  # "fft" here is the Fast Fourier Transform (optimized algorithm)

# Step 6.2: Generate frequency axis (fft_freq) → CRITICAL for interpreting FFT results!
# np.fft.fftfreq(n, d) computes the frequencies corresponding to FFT output bins
#   - n: Length of the signal (number of samples, here N = len(signal))
#   - d: Sampling period (time between consecutive samples, in seconds) = 1/fs
#   Formula for frequency of bin k: f_k = k / (n * d) for k = 0, 1, ..., n-1
#   Note: FFT returns a symmetric spectrum (positive + negative frequencies)
fft_freq = np.fft.fftfreq(N, 1/fs)  # Your original line – now with deep explanation!

# ------------------------------
# 7. Interpret FFT Results
# ------------------------------
# Step 7.1: Extract positive frequency components (key for real-world signals)
# Real signals have hermitian symmetry (negative frequencies are complex conjugates of positive ones)
# Thus, we focus only on positive frequencies (0 Hz to Nyquist Frequency)
nyquist_freq = fs / 2  # Maximum resolvable frequency (here 500 Hz)
positive_mask = fft_freq >= 0  # Boolean mask: True for positive frequencies
positive_freq = fft_freq[positive_mask]  # Extract positive frequencies (0 to 500 Hz)
positive_fft = fft_values[positive_mask]  # Extract corresponding FFT values

# Step 7.2: Scale amplitudes to match true signal amplitude
# FFT amplitudes are proportional but not directly the signal's true amplitude. Scaling is needed:
# - Without scaling: Amplitude = |FFT| / N (FFT sums sample contributions, so divide by total samples)
# - For real signals: positive FFT amplitudes should be doubled (energy split between positive/negative freqs)
# Exception: DC component (0 Hz) – no negative counterpart, so not doubled
amplitude_positive = np.abs(positive_fft) / N * 2  # Double to recover true amplitude
amplitude_positive[0] = amplitude_positive[0] / 2  # Correct DC component (no negative twin)

# Step 7.3: Optional – Reduce noise with a window function (useful for non-periodic signals)
# Windowing minimizes "spectral leakage" (smearing of frequency peaks)
# Example: Hann window (commented out; uncomment to test)
# window = np.hanning(len(amplitude_positive))  # Create Hann window
# amplitude_positive = amplitude_positive * window  # Apply window to smooth noise

# ------------------------------
# 8. Plot the Frequency-Domain Spectrum
# ------------------------------
plt.subplot(2, 1, 2)  # Second plot (bottom)

# Plot frequency vs. scaled amplitude
plt.plot(positive_freq, amplitude_positive, label='FFT Amplitudes (Scaled)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Analysis via FFT (Frequency Domain)')
plt.xlim(0, nyquist_freq)  # Limit x-axis to 0–Nyquist (0–500 Hz)
plt.ylim(0, 2.5)  # Y-axis range to show peaks clearly (max expected amplitude ~2.0 + 1.0)
plt.legend()
plt.grid(True, linestyle='--')  # Grid for readability

# ------------------------------
# 9. Display and Save Results
# ------------------------------
plt.tight_layout()  # Auto-adjust spacing to prevent overlapping plots
plt.show()  # Render the figure

# Optional: Save data for later use (reproducibility or further analysis)
# np.save('fft_freq.npy', positive_freq)
# np.save('fft_amplitude.npy', amplitude_positive)