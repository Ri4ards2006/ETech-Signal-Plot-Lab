"""Butterworth vs. Chebyshev I responses and filtering a noisy test signal.

    uv run python -m examples.filters_demo
    uv run python -m examples.filters_demo --save docs/img
"""

import matplotlib.pyplot as plt
import numpy as np

from examples._cli import run
from signal_lab import filters

FS = 8000.0


def make_figures():
    fig1, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    cases = [("low", 1000.0), ("high", 1000.0), ("band", (500.0, 1500.0))]
    for ax, (btype, edge) in zip(axes, cases):
        for kind in ("butter", "cheby1"):
            sos = filters.design(kind, btype, 5, edge, FS, ripple_db=1.0)
            filters.plot_response(sos, FS, label=f"{kind}, N=5", ax=ax)
        ax.set_title(f"{btype}-pass, edge {edge} Hz")
        ax.set_ylim(-80, 5)
    fig1.tight_layout()

    t = np.arange(int(0.1 * FS)) / FS
    clean = np.sin(2 * np.pi * 50 * t)
    rng = np.random.default_rng(0)
    noisy = clean + 0.5 * np.sin(2 * np.pi * 1500 * t) + 0.2 * rng.standard_normal(t.size)
    sos = filters.design("butter", "low", 6, 200.0, FS)
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1e3, noisy, color="0.7", lw=0.8, label="50 Hz + 1.5 kHz + noise")
    ax.plot(t * 1e3, filters.apply(sos, noisy, zero_phase=False), lw=1.4,
            label="Butterworth LP 200 Hz, causal (sosfilt)")
    ax.plot(t * 1e3, filters.apply(sos, noisy), lw=1.4, label="same, zero-phase (sosfiltfilt)")
    ax.plot(t * 1e3, clean, "k--", lw=1, label="clean 50 Hz")
    ax.set_xlabel("t [ms]")
    ax.set_title("Applying the filter to a test signal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    return {"filters_response.png": fig1, "filters_apply.png": fig2}


if __name__ == "__main__":
    run(make_figures, __doc__)
