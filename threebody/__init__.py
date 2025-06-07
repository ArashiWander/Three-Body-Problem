"""Three-body simulation utilities."""

from importlib.metadata import version, PackageNotFoundError

from .physics import Body, perform_rk4_step, system_energy
from .integrators import compute_accelerations
from .constants import G_REAL, SPACE_SCALE, SOFTENING_FACTOR_SQ
from .simulation import Simulation

try:
    __version__ = version("threebody")
except PackageNotFoundError:  # package is not installed
    __version__ = "0.0.0"

__all__ = [
    'Body',
    'perform_rk4_step',
    'system_energy',
    'compute_accelerations',
    'G_REAL',
    'SPACE_SCALE',
    'Simulation',
    'SOFTENING_FACTOR_SQ',
    '__version__',
]
