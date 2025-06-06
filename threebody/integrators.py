"""Shared numerical integrators for the N-body simulation.

The functions in this module operate purely on NumPy arrays so they can be
reused by both the lightweight physics utilities and the interactive Pygame
simulation.  The implementations are vectorized for clarity and performance.
"""

from __future__ import annotations

import numpy as np

from .constants import G_REAL, SOFTENING_FACTOR_SQ, SPACE_SCALE


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float = G_REAL,
) -> np.ndarray:
    """Return accelerations for each body.

    Parameters
    ----------
    positions : (N, 2) array
        Current positions in simulation units.
    masses : (N,) array
        Body masses.
    fixed_mask : (N,) boolean array
        True for bodies that should not move.
    g_constant : float, optional
        Gravitational constant to use.

    Notes
    -----
    Bodies occupying the exact same position are ignored to avoid numerical
    instability.  Fixed bodies contribute to the field but experience no
    acceleration themselves.
    """
    n = len(masses)
    if n == 0:
        return np.zeros((0, 2), dtype=float)

    # Pairwise displacement vectors r_j - r_i for all i, j
    disp = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist_sq = np.einsum("ijk,ijk->ij", disp, disp)

    # Avoid singularities for self interaction or zero separation
    np.fill_diagonal(dist_sq, np.inf)
    zero_mask = dist_sq == 0.0
    dist_sq[zero_mask] = np.inf

    inv_dist = 1.0 / np.sqrt(dist_sq)
    factors = g_constant * masses[np.newaxis, :] / (
        dist_sq * SPACE_SCALE ** 2 + SOFTENING_FACTOR_SQ
    )

    # acceleration contributions along displacement vectors
    contrib = (factors * inv_dist)[:, :, np.newaxis] * disp
    acc = contrib.sum(axis=1)

    # Fixed bodies experience no acceleration
    acc[fixed_mask] = 0.0

    return acc


def rk4_step_arrays(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    dt: float,
    g_constant: float = G_REAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance bodies using an RK4 step operating on arrays."""

    def deriv(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        return compute_accelerations(pos, masses, fixed_mask, g_constant)

    k1a = deriv(positions, velocities)
    k1v = dt * k1a
    k1p = dt * velocities / SPACE_SCALE

    pos_k2 = positions + 0.5 * k1p
    vel_k2 = velocities + 0.5 * k1v
    k2a = deriv(pos_k2, vel_k2)
    k2v = dt * k2a
    k2p = dt * (velocities + 0.5 * k1v) / SPACE_SCALE

    pos_k3 = positions + 0.5 * k2p
    vel_k3 = velocities + 0.5 * k2v
    k3a = deriv(pos_k3, vel_k3)
    k3v = dt * k3a
    k3p = dt * (velocities + 0.5 * k2v) / SPACE_SCALE

    pos_k4 = positions + k3p
    vel_k4 = velocities + k3v
    k4a = deriv(pos_k4, vel_k4)
    k4v = dt * k4a
    k4p = dt * (velocities + k3v) / SPACE_SCALE

    new_pos = positions.copy()
    new_vel = velocities.copy()
    for i in range(len(masses)):
        if fixed_mask[i]:
            continue
        new_pos[i] += (k1p[i] + 2 * k2p[i] + 2 * k3p[i] + k4p[i]) / 6.0
        new_vel[i] += (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6.0

    return new_pos, new_vel


def rk4_step_bodies(bodies, dt: float, g_constant: float = G_REAL) -> None:
    """Integrate a list of body objects in place using RK4."""
    positions = np.array([b.pos for b in bodies])
    velocities = np.array([b.vel for b in bodies])
    masses = np.array([b.mass for b in bodies])
    fixed_mask = np.array([getattr(b, "fixed", False) for b in bodies])

    new_pos, new_vel = rk4_step_arrays(positions, velocities, masses, fixed_mask, dt, g_constant)

    for b, p, v, fixed in zip(bodies, new_pos, new_vel, fixed_mask):
        if not fixed:
            b.pos = p
            b.vel = v
