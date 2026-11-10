"""FFT spectrum with selectable window and a Fourier-series (Gibbs) demo."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

WINDOWS = {"rect": "boxcar", "hann": "hann", "blackman": "blackman"}

# floor for dB conversion so empty bins give a finite value instead of -inf
_DB_FLOOR = 1e-12


def make_window(name: str, n: int) -> np.ndarray:
    """Periodic (DFT-even) window of length ``n``: 'rect', 'hann' or 'blackman'."""
    try:
        return get_window(WINDOWS[name], n, fftbins=True)
    except KeyError:
        raise ValueError(f"unknown window {name!r}, choose from {sorted(WINDOWS)}") from None


def spectrum(
    x: np.ndarray,
    fs: float,
    window: str = "hann",
    db: bool = True,
    ref: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of a real signal.

    Returns ``(freqs, mag)``. ``freqs`` runs from 0 to fs/2 in steps of fs/N.
    ``mag`` is the peak amplitude of each sinusoidal component, corrected for
    the window's coherent gain, so a bin-centred sine of amplitude A reads A
    (or 20·log10(A/ref) dB) for every window. DC and the Nyquist bin are not
    doubled.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    w = make_window(window, n)
    X = np.fft.rfft(x * w)
    mag = np.abs(X) / w.sum()
    mag[1:] *= 2.0
    if n % 2 == 0:
        mag[-1] /= 2.0  # Nyquist bin has no mirrored partner
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if db:
        mag = 20.0 * np.log10(np.maximum(mag, _DB_FLOOR) / ref)
    return freqs, mag


def square_wave_series(
    t: np.ndarray, f0: float, n_harmonics: int, amplitude: float = 1.0
) -> np.ndarray:
    """Partial Fourier sum of a ±amplitude square wave using the first
    ``n_harmonics`` odd harmonics: 4A/π · Σ sin(2π(2k−1)f0 t)/(2k−1)."""
    t = np.asarray(t, dtype=float)
    k = 2 * np.arange(1, n_harmonics + 1) - 1  # 1, 3, 5, ...
    terms = np.sin(2 * np.pi * f0 * np.multiply.outer(t, k)) / k
    return 4 * amplitude / np.pi * terms.sum(axis=-1)


def gibbs_overshoot(n_harmonics: int) -> float:
    """Overshoot of the partial sum above the plateau, as a fraction of the
    jump height 2A. Tends to Si(π)/π − 1/2 ≈ 0.0895 for many harmonics."""
    # first maximum after the jump sits at ω0·t = π/(2N) for N odd harmonics
    t_peak = 1.0 / (4.0 * n_harmonics)
    t = np.linspace(0.0, 2.0 * t_peak, 20001)
    peak = square_wave_series(t, f0=1.0, n_harmonics=n_harmonics).max()
    return (peak - 1.0) / 2.0


def plot_spectrum(freqs, mag, ax=None, label=None, db=True):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, mag, label=label, lw=1.2)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude [dB]" if db else "Amplitude")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax.figure
