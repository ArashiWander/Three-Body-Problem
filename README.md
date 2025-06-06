# Three-Body Problem Simulation

This repository provides an interactive N-body simulator built with Pygame.  The
core physics is packaged in `threebody` so it can also be reused from other
Python scripts.

## Installation

Install the package and its required dependencies:

```bash
pip install .
```

Numba is used for optional JIT acceleration. If it is missing or fails to
initialize, the simulation automatically falls back to pure Python code so all
features remain available.

## Running the Simulation

Start the interactive application with:

```bash
python threebody/simulation_full.py
```

The Pygame GUI lets you load presets, add new bodies and tweak the simulation
speed or gravitational constant.  Keyboard shortcuts are listed in the in-app
help window.

## Library Usage

Physics helpers can also be imported programmatically:

```python
from threebody import Body, perform_rk4_step, compute_accelerations, system_energy
```

`compute_accelerations` operates directly on NumPy arrays and is shared by both
the lightweight `threebody.physics` module and the interactive simulation.
`perform_rk4_step` advances bodies using a Runge–Kutta 4th order integrator.

The interactive application uses a richer `Body` implementation found in
`threebody.rendering` which adds drawing and trail management utilities.

## Adjusting Body Trails

Use `Body.set_trail_length(length)` to change how many trail points a body
retains. The length is clamped between `MIN_TRAIL_LENGTH` and
`MAX_TRAIL_LENGTH` from `threebody.constants`. Newly created bodies use
`DEFAULT_TRAIL_LENGTH` as the initial value.

## Collision Behaviour

`MERGE_ON_COLLISION` controls what happens when two bodies collide. If set to
`True` the bodies combine into one mass; otherwise they bounce using a simple
elastic collision model.

## Tests

Install the requirements and run the unit tests with:

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

## License

This project is licensed under the [MIT License](LICENSE).
