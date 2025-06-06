# Three-Body Problem Simulation

This project contains a pygame based demonstration of an N-body gravitational system.
A minimal `threebody` package is provided with physics utilities and a full
pygame application is available in `threebody/simulation_full.py`.

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
