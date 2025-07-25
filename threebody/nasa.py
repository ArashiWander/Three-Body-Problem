"""NASA JPL ephemeris helpers."""

from datetime import datetime
from typing import Optional
from pathlib import Path
from urllib.request import urlopen
import shutil

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


def download_ephemeris(url: str, dest: str | Path) -> Path:
    """Download a JPL ephemeris BSP file.

    Parameters
    ----------
    url:
        HTTP(S) location of the BSP file.
    dest:
        Destination directory or full file path where the kernel will be
        written. If ``dest`` is a directory, the filename is taken from
        ``url``.

    Returns
    -------
    Path
        Path of the downloaded file.
    """
    dest_path = Path(dest)
    if dest_path.is_dir():
        dest_path = dest_path / Path(url).name

    with urlopen(url) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    return dest_path.resolve()


def main(argv: list[str] | None = None) -> None:
    """Command line entry point for downloading ephemerides."""
    import argparse

    parser = argparse.ArgumentParser(description="Download a JPL ephemeris BSP file")
    parser.add_argument("url", help="URL of the BSP file")
    parser.add_argument(
        "dest",
        nargs="?",
        default=".",
        help="Destination directory or file path",
    )
    args = parser.parse_args(argv)

    path = download_ephemeris(args.url, args.dest)
    print(path)


if __name__ == "__main__":  # pragma: no cover - manual tool
    main()
