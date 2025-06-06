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

## Adjusting Body Trails

The simulation records the on-screen position of each body so a trail can be
drawn behind it. Trail sizes are clamped between the constants `MIN_TRAIL_LENGTH`
and `MAX_TRAIL_LENGTH` defined in `threebody/constants.py`. When creating a
`Body` you can pass `max_trail_length` or later call
`Body.set_trail_length(new_length)` to change it. The default value is
`DEFAULT_TRAIL_LENGTH`.

## Collision Behaviour

`MERGE_ON_COLLISION` controls what happens when two bodies collide. If set to
`True` the bodies combine into one mass; otherwise they bounce using a simple
elastic collision model.

## Tests and Linting

Install the dependencies and run unit tests with:

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

Linting is performed with `flake8` (install with `pip install flake8`):

```bash
flake8
```

## License

This project is licensed under the [MIT License](LICENSE).

