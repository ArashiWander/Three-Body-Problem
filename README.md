# Three-Body Problem Simulation

This project contains a pygame based demonstration of an N-body gravitational system.
A minimal `threebody` package is provided with physics utilities and a full
pygame application is available in `threebody/simulation_full.py`.

## Installation

Install the package and its dependencies using `pip`:

```bash
pip install .
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



## Collision Behaviour

`MERGE_ON_COLLISION` controls what happens when two bodies collide. If set to
`True` the bodies combine into one mass; otherwise they bounce using a simple
elastic collision model.

## Tests and Linting

Install the dependencies first. These are heavy packages and may take a while to
download and compile:

```bash
pip install -r requirements.txt
```

Run the tests using one of the following options:

1. Without installing the package:

   ```bash
   PYTHONPATH=. pytest -q
   ```

2. Install in editable mode first:

   ```bash
   pip install -e .
   pytest -q
   ```

## License

This project is licensed under the [MIT License](LICENSE).

