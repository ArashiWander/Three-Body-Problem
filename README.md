# Three-Body Problem Simulation

This project contains a pygame based demonstration of an N-body gravitational system.
A minimal `threebody` package is provided with physics utilities and a full
pygame application is available in `threebody/simulation_full.py`.

## Project Structure

The package is organized into focused modules:

- `constants.py` – simulation constants and configurable parameters
- `rendering.py` – the `Body` class and drawing helpers
- `physics_utils.py` – RK4 integration and collision handling
- `jit.py` – optional boundary collision routines accelerated by Numba
- `presets.py` – collections of starting body configurations
- `utils.py` – helpers for formatting values for display

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Simulation

```
python threebody/simulation_full.py
```

### Controls

- **Delete key:** remove the currently selected body.
- **Trail Length slider:** adjust how long body trails are drawn.
- **Merge button:** toggle whether collisions bounce or merge bodies.

The package also exposes physics helpers that can be imported from Python:

```python
from threebody import Body, perform_rk4_step, system_energy
```

## Tests

Unit tests can be executed with `pytest`:

```
pytest
```
