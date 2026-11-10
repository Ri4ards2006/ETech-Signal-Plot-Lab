"""Bode plots of passive RC / RL / RLC networks from component values.

Transfer functions are rational in s and stored as polynomial coefficient
lists (highest power first), as used by ``scipy.signal.freqs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import freqs

HALF_POWER_DB = 10 * np.log10(2)  # 3.0103 dB


@dataclass(frozen=True)
class TransferFunction:
    num: tuple[float, ...]
    den: tuple[float, ...]
    name: str = ""

    def __call__(self, f):
        """Complex H(j2πf)."""
        w = 2 * np.pi * np.atleast_1d(np.asarray(f, dtype=float))
        return freqs(self.num, self.den, worN=w)[1]


def _positive(**values):
    for k, v in values.items():
        if not v > 0:
            raise ValueError(f"{k} must be > 0, got {v!r}")


def rc_lowpass(R, C):
    """Series R, output across C: H = 1 / (1 + sRC)."""
    _positive(R=R, C=C)
    return TransferFunction((1.0,), (R * C, 1.0), f"RC low-pass R={R:g} Ω, C={C:g} F")


def rc_highpass(R, C):
    """Series C, output across R: H = sRC / (1 + sRC)."""
    _positive(R=R, C=C)
    return TransferFunction((R * C, 0.0), (R * C, 1.0), f"RC high-pass R={R:g} Ω, C={C:g} F")


def rl_lowpass(R, L):
    """Series L, output across R: H = R / (R + sL)."""
    _positive(R=R, L=L)
    return TransferFunction((R,), (L, R), f"RL low-pass R={R:g} Ω, L={L:g} H")


def rl_highpass(R, L):
    """Series R, output across L: H = sL / (R + sL)."""
    _positive(R=R, L=L)
    return TransferFunction((L, 0.0), (L, R), f"RL high-pass R={R:g} Ω, L={L:g} H")


def rlc_bandpass(R, L, C):
    """Series RLC, output across R: H = sRC / (s²LC + sRC + 1)."""
    _positive(R=R, L=L, C=C)
    return TransferFunction(
        (R * C, 0.0), (L * C, R * C, 1.0), f"RLC band-pass R={R:g} Ω, L={L:g} H, C={C:g} F"
    )


def rlc_lowpass(R, L, C):
    """Series RLC, output across C: H = 1 / (s²LC + sRC + 1)."""
    _positive(R=R, L=L, C=C)
    return TransferFunction(
        (1.0,), (L * C, R * C, 1.0), f"RLC low-pass R={R:g} Ω, L={L:g} H, C={C:g} F"
    )


def response(tf: TransferFunction, f) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude [dB] and phase [deg] at frequencies ``f`` [Hz]."""
    h = tf(f)
    return 20 * np.log10(np.abs(h)), np.degrees(np.angle(h))


def _frequency_span(tf: TransferFunction) -> tuple[float, float]:
    roots = np.concatenate([np.roots(tf.num), np.roots(tf.den)])
    corners = np.abs(roots[np.abs(roots) > 0]) / (2 * np.pi)
    return corners.min() / 1e4, corners.max() * 1e4


def _reference_gain(tf: TransferFunction, f_lo: float, f_hi: float) -> float:
    """Pass-band gain.

    Low-/high-pass: the asymptotic gain |H(0)| or |H(∞)| (so a resonant
    low-pass is referenced to DC, not to its peak). Band-pass (both
    asymptotes zero): the peak gain, located numerically.
    """
    num, den = np.asarray(tf.num), np.asarray(tf.den)
    g0 = abs(num[-1] / den[-1]) if den[-1] != 0 else np.inf
    ginf = abs(num[0] / den[0]) if len(num) == len(den) else 0.0
    asymptotic = max(g0, ginf)
    if asymptotic > 0:
        return asymptotic
    grid = np.geomspace(f_lo, f_hi, 4001)
    k = int(np.argmax(np.abs(tf(grid))))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    res = minimize_scalar(
        lambda lf: -abs(tf(10**lf)[0]), bounds=(np.log10(lo), np.log10(hi)), method="bounded",
        options={"xatol": 1e-12},
    )
    return -res.fun


def minus_3db_frequencies(tf: TransferFunction) -> list[float]:
    """All frequencies where |H| is 3.01 dB (half power) below the pass-band gain."""
    f_lo, f_hi = _frequency_span(tf)
    ref = _reference_gain(tf, f_lo, f_hi)

    def g(lf):
        return abs(tf(10**lf)[0]) ** 2 - ref**2 / 2

    lg = np.linspace(np.log10(f_lo), np.log10(f_hi), 4001)
    vals = np.abs(tf(10**lg)) ** 2 - ref**2 / 2
    idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    return [10 ** brentq(g, lg[i], lg[i + 1], xtol=1e-14) for i in idx]


def plot_bode(tf: TransferFunction, f=None, mark_3db=True):
    """Two-panel Bode plot (magnitude + phase) with −3 dB markers."""
    import matplotlib.pyplot as plt

    if f is None:
        f_lo, f_hi = _frequency_span(tf)
        f = np.geomspace(f_lo * 1e2, f_hi / 1e2, 1000)
    mag, phase = response(tf, f)
    fig, (am, ap) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    am.semilogx(f, mag, lw=1.5)
    ap.semilogx(f, phase, lw=1.5)
    if mark_3db:
        corners = minus_3db_frequencies(tf)
        for i, fc in enumerate(corners):
            m, p = response(tf, [fc])
            for ax, y in ((am, m[0]), (ap, p[0])):
                ax.axvline(fc, color="r", ls="--", lw=0.8)
                ax.plot(fc, y, "ro")
            left = i == 0 and len(corners) > 1  # lower corner labelled to the left
            am.annotate(f"−3 dB @ {fc:,.1f} Hz", (fc, m[0]), textcoords="offset points",
                        xytext=(-6 if left else 6, -14), ha="right" if left else "left",
                        color="r", fontsize=9)
    am.set_ylabel("|H| [dB]")
    ap.set_ylabel("phase [°]")
    ap.set_xlabel("Frequency [Hz]")
    am.set_title(tf.name)
    for ax in (am, ap):
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig
