"""Step and impulse responses of 1st- and 2nd-order LTI systems.

    1st order:  G(s) = K / (τs + 1)
    2nd order:  G(s) = K·ωn² / (s² + 2ζωn·s + ωn²)

Numerical responses come from ``scipy.signal`` (state-space, matrix
exponential); the closed forms below are used to verify them.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def first_order(K: float, tau: float) -> signal.lti:
    if not tau > 0:
        raise ValueError(f"tau must be > 0, got {tau!r}")
    return signal.lti([K], [tau, 1.0])


def second_order(zeta: float, wn: float, K: float = 1.0) -> signal.lti:
    if not zeta >= 0:
        raise ValueError(f"zeta must be >= 0, got {zeta!r}")
    if not wn > 0:
        raise ValueError(f"wn must be > 0, got {wn!r}")
    return signal.lti([K * wn**2], [1.0, 2 * zeta * wn, wn**2])


def step(sys: signal.lti, t):
    return signal.step(sys, T=np.asarray(t, dtype=float))


def impulse(sys: signal.lti, t):
    return signal.impulse(sys, T=np.asarray(t, dtype=float))


# --- closed forms -----------------------------------------------------------

def first_order_step(t, K, tau):
    return K * (1 - np.exp(-np.asarray(t) / tau))


def first_order_impulse(t, K, tau):
    return K / tau * np.exp(-np.asarray(t) / tau)


def second_order_step(t, zeta, wn, K=1.0):
    t = np.asarray(t, dtype=float)
    if zeta < 1:  # under-damped (incl. undamped ζ = 0)
        wd = wn * np.sqrt(1 - zeta**2)
        y = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta**2) * np.sin(wd * t))
    elif zeta == 1:  # critically damped
        y = 1 - np.exp(-wn * t) * (1 + wn * t)
    else:  # over-damped: two real poles s1, s2
        r = np.sqrt(zeta**2 - 1)
        s1, s2 = -wn * (zeta - r), -wn * (zeta + r)
        y = 1 + (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s1 - s2)
    return K * y


def second_order_impulse(t, zeta, wn, K=1.0):
    t = np.asarray(t, dtype=float)
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta**2)
        h = wn / np.sqrt(1 - zeta**2) * np.exp(-zeta * wn * t) * np.sin(wd * t)
    elif zeta == 1:
        h = wn**2 * t * np.exp(-wn * t)
    else:
        r = np.sqrt(zeta**2 - 1)
        s1, s2 = -wn * (zeta - r), -wn * (zeta + r)
        h = wn**2 * (np.exp(s1 * t) - np.exp(s2 * t)) / (s1 - s2)
    return K * h


# --- characteristic values --------------------------------------------------

def overshoot_percent(zeta: float) -> float:
    """Peak overshoot of the 2nd-order step response: 100·exp(−πζ/√(1−ζ²))."""
    if zeta >= 1:
        return 0.0
    return 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))


def peak_time(zeta: float, wn: float) -> float:
    """Time of the first maximum: π / ωd (under-damped only)."""
    if zeta >= 1:
        return np.inf
    return np.pi / (wn * np.sqrt(1 - zeta**2))


def _first_crossing(t, y, level):
    i = int(np.argmax(y >= level))
    if i == 0:
        return t[0]
    return np.interp(level, [y[i - 1], y[i]], [t[i - 1], t[i]])


def step_metrics(t, y, final=None, band=0.02) -> dict:
    """Overshoot [%], peak time, 10–90 % rise time and settling time (±band)
    measured from a sampled step response. ``final`` defaults to y[-1]."""
    t, y = np.asarray(t), np.asarray(y)
    final = y[-1] if final is None else final
    k = int(np.argmax(y))
    err, tol = np.abs(y - final), band * abs(final)
    outside = np.nonzero(err > tol)[0]
    if outside.size == 0:
        settling = t[0]
    elif outside[-1] + 1 < t.size:
        i = outside[-1]  # last sample outside the band: interpolate the exit
        settling = np.interp(tol, [err[i + 1], err[i]], [t[i + 1], t[i]])
    else:
        settling = np.inf
    return {
        "overshoot_pct": max(0.0, 100 * (y[k] - final) / final),
        "peak_time": t[k],
        "rise_time": _first_crossing(t, y, 0.9 * final) - _first_crossing(t, y, 0.1 * final),
        "settling_time": settling,
    }


def plot_family(zetas, wn=1.0, t=None, kind="step"):
    import matplotlib.pyplot as plt

    t = np.linspace(0, 15 / wn, 2000) if t is None else t
    f = second_order_step if kind == "step" else second_order_impulse
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for z in zetas:
        ax.plot(t * wn, f(t, z, wn), lw=1.4, label=f"ζ = {z:g}")
    ax.set_xlabel("ωn·t")
    ax.set_ylabel("y(t)" if kind == "step" else "h(t)")
    ax.set_title(f"2nd-order {kind} response, G(s) = ωn² / (s² + 2ζωn s + ωn²)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig
