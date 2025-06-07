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
Installing `cupy` enables optional GPU support.

CuPy can be installed to run the core gravitational calculation on a CUDA GPU.
Install a build matching your CUDA toolkit, for example:

```bash
pip install cupy-cuda12x
```

With CuPy available and a compatible GPU present, `compute_accelerations`
automatically performs its work on the GPU. If no GPU is detected it silently
falls back to the CPU.

## Running the Simulation

Start the interactive application with:

```bash
python -m threebody.simulation_full
```

Use `--softening-length` to override the gravitational softening length in
metres when launching the simulation.
Pass ``--use-gpu`` to offload calculations to a CUDA capable device when the
optional CuPy dependency is installed.

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

## Integrator Choices

Three integrators are provided:

* **Fixed Step RK4** via :py:meth:`perform_rk4_step` – classical fourth-order
  Runge–Kutta integration.
* **Adaptive RK4** via :py:meth:`threebody.physics_utils.adaptive_rk4_step` –
  automatically adjusts the time step to keep the estimated local error below
  ``ERROR_TOLERANCE``.
* **Symplectic Leapfrog** – a new energy conserving integrator available from
  ``threebody.integrators``. It is particularly suited to long term orbital
  integrations.

These are standard explicit methods as described in
[Hairer et&nbsp;al.](https://doi.org/10.1007/978-3-642-05415-1) and
[Leimkuhler &amp; Reich, 2004](https://doi.org/10.1017/CBO9780511614117) for the
symplectic scheme.

In the interactive simulation you can toggle adaptive stepping at runtime by
pressing <kbd>A</kbd>. A ``--use-gpu`` flag enables GPU acceleration when a
compatible CUDA device and the optional CuPy package are available.

## Unit Conventions

Positions are stored in **simulation units** where one unit equals
`SPACE_SCALE` metres (defined in `threebody.constants`).  Velocities and time
steps are always specified in SI metres per second and seconds respectively
while masses are given in kilograms.  When creating bodies programmatically be
sure to convert distances from metres to simulation units:

```python
from threebody import SPACE_SCALE
x_sim = real_distance_meters / SPACE_SCALE
```

Using unscaled metre values will make gravity appear far too weak because the
distances become millions of times larger than intended.  The same scale is used
internally for presets so that real solar system parameters remain numerically
stable.

Alternatively, call :py:meth:`Body.from_meters` to create a body using SI
coordinates directly. The helper converts the position for you so the resulting
objects interact as expected.  This is important because the gravitational
constant ``G_REAL`` bundled with the library is expressed in SI units.

## Adjusting Body Trails

Use `Body.set_trail_length(length)` to change how many trail points a body
retains. The length is clamped between `MIN_TRAIL_LENGTH` and
`MAX_TRAIL_LENGTH` from `threebody.constants`. Newly created bodies use
`DEFAULT_TRAIL_LENGTH` as the initial value.

## Collision Behaviour

`MERGE_ON_COLLISION` controls what happens when two bodies collide. If set to
`True` the bodies combine into one mass; otherwise they bounce using a simple
elastic collision model.

## Configuration Parameters

The `threebody.constants` module exposes numerous tuning options. Important
values include:

* ``SPACE_SCALE`` – number of metres represented by one simulation unit.
* ``TIME_STEP_BASE`` – default integration step size in seconds.
* ``ERROR_TOLERANCE`` – target accuracy for adaptive stepping.
* ``SOFTENING_LENGTH`` – softening length in metres used to avoid numerical
  singularities. Override this at runtime with ``--softening-length`` when
  launching ``simulation_full``.

## Simulation Accuracy

Both integrators are fourth‑order Runge–Kutta schemes which provide good
accuracy for moderate step sizes.  Unit tests integrate the Earth–Sun system for
30 days and verify that total energy is conserved to within 0.1%, confirming the
stability of the default parameters.

Gravitational forces follow Newton's law with a small softening term to avoid
singularities.  Numerical errors will grow over time, so reducing the time step
or enabling adaptive stepping improves long‑term stability.

## Reproducing Benchmark Scenarios

Several presets bundled with the app allow you to verify the simulator against
well known solutions.  Select **Figure 8** from the preset menu to reproduce the
periodic three‑body orbit described by [Chenciner & Montgomery, 2000](https://doi.org/10.1007/s002200050016).
To reproduce the automated benchmarks used during development run

```bash
pip install -r requirements.txt
NUMBA_DISABLE_JIT=1 PYTHONPATH=. pytest -k benchmark -q
```

This executes a scenario where the Earth orbits the Sun for 30 days and checks
that the energy drift stays below 0.1% using the symplectic integrator. The
benchmark definitions live in ``tests/test_energy.py`` and can be modified to
experiment with alternative parameters.

## Tests

Install the requirements and run the unit tests with:

```bash
pip install -r requirements.txt
NUMBA_DISABLE_JIT=1 PYTHONPATH=. pytest -q
```

Disabling numba's JIT avoids startup issues on systems where the bundled
llvmlite does not support the host CPU's features.

## Scientific References

* Newton's law of gravitation – *Philosophiæ Naturalis Principia Mathematica*, 1687.
* E. Hairer, S. P. Nørsett &amp; G. Wanner, *Solving Ordinary Differential
  Equations I*, 2nd edition, Springer, 2008.
* B. Leimkuhler &amp; S. Reich, *Simulating Hamiltonian Dynamics*, Cambridge
  University Press, 2004.
* A. Chenciner &amp; R. Montgomery, "A remarkable periodic solution of the
  three-body problem in the case of equal masses," *Annals of Mathematics*,
  152(3), 2000.

## License

This project is licensed under the [MIT License](LICENSE).
