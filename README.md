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

The package also exposes physics helpers that can be imported from Python:

```python
from threebody import Body, perform_rk4_step, system_energy
```

## Tests

Unit tests can be executed with `pytest`:

```
pytest
```

### Linting

Run `flake8` to check code style:

```
flake8
```

## Simulation Options

- `MIN_TRAIL_LENGTH` and `MAX_TRAIL_LENGTH` limit the number of trail points stored for each body. Use the UI slider in the "Selected Body" section to adjust the trail length for that body.
- `MERGE_ON_COLLISION` toggles whether colliding bodies bounce or merge. This constant is defined in `threebody/constants.py`.

This project is released under the MIT License. See LICENSE for details.
