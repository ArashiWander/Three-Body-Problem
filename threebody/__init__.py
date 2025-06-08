"""Three-body simulation utilities."""

from importlib.metadata import PackageNotFoundError, version

from .physics import Body, perform_rk4_step, system_energy
from .integrators import compute_accelerations
from .constants import (
    G_REAL,
    SPACE_SCALE,
    SOFTENING_LENGTH,
    SOFTENING_FACTOR_SQ,
    C_LIGHT,
)

from .nasa import load_ephemeris, body_state, create_body, download_ephemeris
from .state_manager import save_state, load_state
try:
    __version__ = version("threebody")
except PackageNotFoundError:
    # Fallback when package metadata is unavailable (e.g. running from source)
    __version__ = "0.0.0"

__all__ = [
    "Body",
    "perform_rk4_step",
    "system_energy",
    "compute_accelerations",
    "G_REAL",
    "SPACE_SCALE",
    "SOFTENING_LENGTH",
    "SOFTENING_FACTOR_SQ",
    "C_LIGHT",
    "__version__",
    "load_ephemeris",
    "body_state",
    "create_body",
    "download_ephemeris",
    "save_state",
    "load_state",
]
