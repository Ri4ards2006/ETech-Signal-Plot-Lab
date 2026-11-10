"""Shared entry point for the example scripts: show figures or save PNGs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def save_or_show(figs: dict, save_dir=None) -> list[Path]:
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


def run(make_figures, doc: str, interactive=None) -> None:
    """``--save DIR`` writes the static figures; otherwise opens ``interactive()``
    (if given, e.g. a slider figure) or the static figures in windows."""
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    ap.add_argument("--save", metavar="DIR", help="write PNGs instead of opening windows")
    save_dir = ap.parse_args().save
    if save_dir is None and interactive is not None:
        _keep = interactive()  # widgets must stay referenced while the window is open
        plt.show()
        return
    save_or_show(make_figures(), save_dir)
