"""Legacy entry point for running the interactive simulation.

This module is kept for backwards compatibility. It simply imports and
executes :class:`Simulation` from :mod:`threebody.simulation`.
"""

from .simulation import Simulation

if __name__ == "__main__":
    Simulation().run()
