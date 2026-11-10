"""FFT window comparison + square wave from odd harmonics (Gibbs).

    uv run python -m examples.fourier_demo            # show windows
    uv run python -m examples.fourier_demo --save docs/img
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from signal_lab import fourier


def make_figures():
    fs, n = 1000.0, 1000
    t = np.arange(n) / fs
    # off-bin tone (50.5 Hz) plus a weak 120 Hz tone 60 dB below
    x = np.sin(2 * np.pi * 50.5 * t) + 1e-3 * np.sin(2 * np.pi * 120 * t)

    fig1, ax = plt.subplots(figsize=(9, 4.5))
    for w in ("rect", "hann", "blackman"):
        f, mag = fourier.spectrum(x, fs, window=w)
        fourier.plot_spectrum(f, mag, ax=ax, label=w)
    ax.set_xlim(0, 250)
    ax.set_ylim(-160, 5)
    ax.set_title("FFT spectrum: 50.5 Hz (0 dB) + 120 Hz (−60 dB), fs = 1 kHz, N = 1000")

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    tt = np.linspace(-0.25, 1.25, 6000)
    ideal = np.sign(np.sin(2 * np.pi * tt))
    a1.plot(tt, ideal, "k--", lw=1, label="ideal")
    for nh in (1, 3, 10, 50):
        a1.plot(tt, fourier.square_wave_series(tt, 1.0, nh), lw=1, label=f"{nh} odd harmonics")
    a1.set_title("Square wave from odd harmonics")
    a1.set_xlabel("t / T")
    a1.legend(fontsize=8, loc="lower left")
    a1.grid(alpha=0.3)

    ns = np.array([1, 2, 3, 5, 10, 20, 50, 100, 200, 500])
    a2.semilogx(ns, [100 * fourier.gibbs_overshoot(k) for k in ns], "o-")
    a2.axhline(8.949, color="r", ls="--", label="Si(π)/π − ½ = 8.95 %")
    a2.set_title("Gibbs overshoot (relative to jump)")
    a2.set_xlabel("number of odd harmonics")
    a2.set_ylabel("overshoot [%]")
    a2.legend()
    a2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    return {"fourier_windows.png": fig1, "fourier_gibbs.png": fig2}


def main(save_dir=None):
    figs = make_figures()
    if save_dir is None:
        plt.show()
        return []
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        fig.savefig(out / name, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(out / name)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", metavar="DIR", help="write PNGs instead of opening windows")
    main(ap.parse_args().save)
