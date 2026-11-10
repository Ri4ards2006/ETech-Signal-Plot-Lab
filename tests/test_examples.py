"""Smoke test: every example builds its figures and saves them as PNG."""

import importlib

import pytest

from examples._cli import save_or_show

EXAMPLES = ["fourier_demo", "bode_demo", "filters_demo", "sampling_demo", "systems_demo"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_saves_png(name, tmp_path):
    mod = importlib.import_module(f"examples.{name}")
    paths = save_or_show(mod.make_figures(), tmp_path)
    assert paths
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_sampling_slider_updates():
    from examples import sampling_demo

    slider = sampling_demo.interactive()
    slider.set_val(150)  # aliased case redraws without error
    assert slider.val == 150
