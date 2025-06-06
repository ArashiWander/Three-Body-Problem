# Three-Body Problem Simulation

This repository provides an interactive N-body simulator built with Pygame.  The
core physics is packaged in `threebody` so it can also be reused from other
Python scripts.

## Installation

Install the package and its required dependencies from the repository root—the
folder containing `pyproject.toml`:

```bash
git clone <repo-url>
cd Three-Body-Problem
pip install .
```

Numba is used for optional JIT acceleration. If it is missing or fails to
initialize, the simulation automatically falls back to pure Python code so all
features remain available.

## Running the Simulation

Start the interactive application with:

```bash
python -m threebody.simulation_full
```

### Quick Start

After launching you will see a control panel on the left. To get moving quickly:

1. Pick a preset from the **Preset** drop‑down (for example *Sun & Earth*).
2. Press **SPACE** to pause or resume the motion.
3. Drag with the **right mouse button** to create a new body – the drag
   direction sets its starting velocity.
4. Use the **Speed** and **Gravity** sliders to tune the simulation.
5. Press **H** or click **Help / Controls** at any time for a full list of
   shortcuts.

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

## Coordinate Units

Positions in `threebody` are expressed in **simulation units**.  One unit
corresponds to `SPACE_SCALE` metres (see `threebody.constants`).  Velocities are
always given in metres per second.  When creating bodies programmatically be
sure to convert distances from metres to simulation units:

```python
from threebody import SPACE_SCALE
x_sim = real_distance_meters / SPACE_SCALE
```

Using unscaled metre values will make gravity appear far too weak because the
distances become millions of times larger than intended.

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
NUMBA_DISABLE_JIT=1 PYTHONPATH=. pytest -q
```

Disabling numba's JIT avoids startup issues on systems where the bundled
llvmlite does not support the host CPU's features.

## License

This project is licensed under the [MIT License](LICENSE).
