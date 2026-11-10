"""Sampling, Nyquist criterion, aliasing and ideal (sinc) reconstruction."""

from __future__ import annotations

import numpy as np


def nyquist_rate(f_max: float) -> float:
    """Minimum sampling rate for a signal band-limited to ``f_max``: 2·f_max."""
    return 2.0 * f_max


def is_aliased(f: float, fs: float) -> bool:
    """True if a tone at ``f`` violates fs > 2f (f = fs/2 counts as aliased:
    a sine exactly at Nyquist can be sampled at its zero crossings)."""
    return fs <= nyquist_rate(f)


def alias_frequency(f, fs):
    """Apparent frequency of a tone ``f`` sampled at ``fs``, folded into [0, fs/2]."""
    f = np.asarray(f, dtype=float)
    return np.abs(f - fs * np.round(f / fs))


def sample(f: float, fs: float, duration: float, amplitude: float = 1.0, phase: float = 0.0):
    """Sample ``amplitude·sin(2πft + phase)``; returns ``(t_n, x_n)``."""
    tn = np.arange(int(round(duration * fs))) / fs
    return tn, amplitude * np.sin(2 * np.pi * f * tn + phase)


def sinc_reconstruct(t, tn, xn, fs):
    """Whittaker–Shannon interpolation x(t) = Σ x[n]·sinc(fs·t − n).

    Exact for band-limited signals and infinite records; with a finite record
    the error is small away from the edges.
    """
    t = np.asarray(t, dtype=float)
    return np.sinc(fs * (t[:, None] - np.asarray(tn)[None, :])) @ np.asarray(xn)
