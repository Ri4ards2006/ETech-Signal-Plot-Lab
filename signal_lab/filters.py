"""Digital IIR filter design (Butterworth / Chebyshev I) via scipy.signal.

Filters are returned as second-order sections (SOS), which stay numerically
stable at high orders where (b, a) polynomials do not.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

KINDS = ("butter", "cheby1")
BTYPES = {"low": "lowpass", "high": "highpass", "band": "bandpass"}


def design(kind, btype, order, cutoff, fs, ripple_db=1.0):
    """Design a digital IIR filter.

    kind:   'butter' (maximally flat, −3.01 dB at ``cutoff``) or
            'cheby1' (pass-band ripple ``ripple_db``, −ripple at ``cutoff``)
    btype:  'low', 'high' or 'band'
    cutoff: edge frequency in Hz, or (f_low, f_high) for 'band'
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if btype not in BTYPES:
        raise ValueError(f"btype must be one of {tuple(BTYPES)}, got {btype!r}")
    edges = np.atleast_1d(np.asarray(cutoff, dtype=float))
    if (btype == "band") != (edges.size == 2):
        raise ValueError("'band' needs (f_low, f_high); 'low'/'high' need one edge")
    if np.any(edges <= 0) or np.any(edges >= fs / 2):
        raise ValueError(f"cutoff must lie in (0, fs/2 = {fs / 2:g}) Hz, got {cutoff!r}")
    wn = edges if btype == "band" else edges[0]
    if kind == "butter":
        return signal.butter(order, wn, BTYPES[btype], fs=fs, output="sos")
    return signal.cheby1(order, ripple_db, wn, BTYPES[btype], fs=fs, output="sos")


def frequency_response(sos, fs, f=None, n=4096):
    """Complex response at frequencies ``f`` [Hz] (default: n points to fs/2)."""
    worN = n if f is None else np.asarray(f, dtype=float)
    return signal.sosfreqz(sos, worN=worN, fs=fs)


def apply(sos, x, zero_phase=True):
    """Filter ``x``. ``zero_phase`` runs forward+backward (no delay, squared
    magnitude); otherwise a causal single pass as a real-time filter would."""
    return signal.sosfiltfilt(sos, x) if zero_phase else signal.sosfilt(sos, x)


def plot_response(sos, fs, label=None, ax=None, n=4096):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    f, h = frequency_response(sos, fs, n=n)
    ax.plot(f, 20 * np.log10(np.maximum(np.abs(h), 1e-12)), label=label, lw=1.4)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|H| [dB]")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax.figure
