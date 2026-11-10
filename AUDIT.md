# AUDIT — ETech-Signal-Plot-Lab (state of `main` @ d326a2b)

## Language / stack

| Part | Stack | Notes |
|---|---|---|
| Python scripts (~3.6 k LOC, 22 files) | NumPy, Matplotlib, SciPy (`signal`, `integrate`), ipywidgets | No `pyproject.toml` / `requirements.txt`; README only says `pip install numpy matplotlib scipy` (ipywidgets missing). |
| `C++/`, `C++/Freq_Analyzer/` | C++23, CMake ≥ 4.0, CLion project files (`.idea/`) committed | Embedded-style frequency analyzer skeleton (RingBuffer, ADCBuffer, Timer, SignalGenerator). |
| `learn.jar` | 156-byte binary | Not a usable artifact. |

## What it plots today

| Area | Files | Content |
|---|---|---|
| Signals | `src/signals/sine_cos.py`, `square_triangle(s).py`, `signal_demo(_v2).py` | Sine/cos, square/triangle, noise; Matplotlib sliders. |
| RLC | `src/signals/RLC_Simulator.py` | ODE (`odeint`) of a series RLC charge/current, ipywidgets sliders. |
| Filters | `circuits/RC_LowPass.py`, `src/filters/tschebyscheff_Filter.py`, `src/filters/Active_Lowpassfilter_Sallen-Key.py` | RC low-pass Bode, Chebyshev I design + Bode comparison, Sallen-Key response. |
| Op-amps | `src/control/Op_Amps_*.py`, `Op_Anos_Komporator.py` | Inverter / comparator transfer and time plots. |
| RF / comms | `comms/HF_Impedance_Matching.py`, `src/comms/Smitch_Chart_Plot.py`, `comms/Spectrum_analyzer.py` | L/π/T matching networks, Smith chart. |
| Antennas | `comms/hf-antenna-pattern.py`, `comms/3D-FieldVisualisation.py` | Dipole / array / Yagi patterns (2D + 3D). |
| Channel / Doppler | `comms/Canal_Effect_Lab.py`, `comms/Doppler_Effect.py` | Channel effects incl. eye diagram, Doppler shift. |

## Structure

```
C++/                 CMake projects (+ committed .idea/)
circuits/            RC_LowPass.py, Filter_Toolkit.py (empty)
comms/               RF/antenna/channel scripts, Modulation/Modulation_Toolkit.py (empty)
src/{signals,filters,control,comms}/   further standalone scripts
README.md, learn.jar
```

Every script is standalone: computation, plotting and `plt.show()` / `interact(...)` run at import time.

## Gaps

1. **No tests, no CI.** Nothing checks the plotted numbers against analytic results.
2. **Not importable.** Module-level `plt.show()` / `interact()` mean nothing can be reused or tested; file names with `-` (`hf-antenna-pattern.py`, `Active_Lowpassfilter_Sallen-Key.py`) cannot be imported at all.
3. **No packaging / dependency pinning.** No `pyproject.toml`; ipywidgets dependency undeclared; scripts need Jupyter for the widget versions.
4. **Duplication / misnaming.** `comms/Spectrum_analyzer.py` is an impedance-matching script, not a spectrum analyzer; the impedance code is copied across three files. Two copies of square/triangle and comparator scripts (`Op_Anos_Komporator.py` typo).
5. **Empty placeholders.** `circuits/Filter_Toolkit.py`, `comms/Modulation/Modulation_Toolkit.py`, C++ `FFT`/`Window` classes are empty — the README advertises Fourier, modulation, Bode, step responses that do not exist yet.
6. **No FFT / windowing / dB spectrum**, no Fourier-series (Gibbs) demo, no sampling/aliasing demo, no generic 1st/2nd-order step/impulse responses, no RL/RLC Bode from component values, no Butterworth filters.
7. **C++ build broken on a clean checkout:** `C++/Freq_Analyzer/CMakeLists.txt` lists `cmake-build-debug/PWMController.{cpp,h}`, which are not tracked.
8. **Repo hygiene:** `.idea/` and `.directory` files committed, no `.gitignore`, `learn.jar` stub.

## Plan for `claude/dsp-modules`

New importable package `signal_lab/` (compute functions return arrays, plot helpers return `Figure`, no `plt.show()` on import), one example script per module in `examples/`, pytest tests that compare against closed-form results, GitHub Actions CI, README with generated plots, `docs/PORTFOLIO.md` (EN + DE). Existing scripts are left untouched; gaps 4, 7, 8 are reported here, not fixed in this branch.
