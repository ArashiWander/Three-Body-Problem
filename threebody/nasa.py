"""NASA JPL ephemeris helpers."""

from datetime import datetime
from typing import Optional

import numpy as np
from jplephem.spk import SPK

from .physics import Body
from .constants import SPACE_SCALE


def load_ephemeris(path: str) -> SPK:
    """Load a JPL SPK ephemeris file."""
    return SPK.open(path)


def body_state(ephem: SPK, target: int, epoch: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Return position and velocity in simulation units and m/s."""
    jd = epoch.timestamp() / 86400.0 + 2440587.5
    pos, vel = ephem[0, target].compute_and_differentiate(jd)
    pos = (pos * 1000.0) / SPACE_SCALE
    vel = vel * 1000.0
    return pos, vel


def create_body(
    ephem: SPK,
    target: int,
    epoch: datetime,
    mass: float,
    *,
    name: Optional[str] = None,
) -> Body:
    """Create a :class:`Body` instance from ephemeris data."""
    pos, vel = body_state(ephem, target, epoch)
    return Body(mass, pos, vel, name=name or str(target))
