"""Utility helpers for unit conversions."""

from . import constants as C


def mass_to_display(mass_kg: float) -> str:
    if mass_kg == 0:
        return "0 kg"
    if mass_kg >= 0.1 * C.SOLAR_MASS:
        return f"{mass_kg/C.SOLAR_MASS:.2f} M☉"
    if mass_kg >= 0.1 * C.EARTH_MASS:
        return f"{mass_kg/C.EARTH_MASS:.2f} M⊕"
    return f"{mass_kg:.2e} kg"


def distance_to_display(dist_meters: float) -> str:
    if dist_meters == 0:
        return "0 m"
    if abs(dist_meters) >= 0.1 * C.AU:
        return f"{dist_meters/C.AU:.2f} AU"
    if abs(dist_meters) >= 1e6:
        return f"{dist_meters/1e6:.2f} Mm"
    if abs(dist_meters) >= 1e3:
        return f"{dist_meters/1e3:.2f} km"
    return f"{dist_meters:.1f} m"


def time_to_display(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    if seconds == 0:
        return "0 sec"
    years = seconds / 31536000
    if years >= 1:
        return f"{years:.1f} years"
    days = seconds / 86400
    if days >= 1:
        return f"{days:.1f} days"
    hours = seconds / 3600
    if hours >= 1:
        return f"{hours:.1f} hrs"
    minutes = seconds / 60
    if minutes >= 1:
        return f"{minutes:.1f} min"
    return f"{seconds:.1f} sec"
