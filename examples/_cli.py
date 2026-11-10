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


def run(make_figures, doc: str) -> None:
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    ap.add_argument("--save", metavar="DIR", help="write PNGs instead of opening windows")
    save_or_show(make_figures(), ap.parse_args().save)
