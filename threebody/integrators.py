from __future__ import annotations

import numpy as np

from . import constants as C
from .jit import nb, NUMBA_AVAILABLE


@nb.njit
def _compute_accelerations_jit(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float,
    space_scale_sq: float,
    softening_sq: float,
) -> np.ndarray:
    n = masses.shape[0]
    dim = positions.shape[1]
    acc = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        if fixed_mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = 0.0
            if dim == 3:
                dz = positions[j, 2] - positions[i, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq == 0.0:
                continue
            dist_sq_m = dist_sq * space_scale_sq
            inv_dist = 1.0 / np.sqrt(dist_sq)
            factor = g_constant * masses[j] / (dist_sq_m + softening_sq)
            acc[i, 0] += dx * inv_dist * factor
            acc[i, 1] += dy * inv_dist * factor
            if dim == 3:
                acc[i, 2] += dz * inv_dist * factor
    return acc


def _compute_accelerations_py(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float,
) -> np.ndarray:
    n = len(masses)
    dim = positions.shape[1]
    acc = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        if fixed_mask[i]:
            continue

        r_vec = positions - positions[i]
        dist_sq = np.einsum("ij,ij->i", r_vec, r_vec)
        dist_sq[i] = np.inf

        dist_sq_m = dist_sq * (C.SPACE_SCALE**2)

        inv_dist = np.zeros_like(dist_sq)
        mask = dist_sq > 0.0
        inv_dist[mask] = 1.0 / np.sqrt(dist_sq[mask])

        factors = g_constant * masses / (dist_sq_m + C.SOFTENING_FACTOR_SQ)

        acc[i] = np.sum(r_vec * (inv_dist[:, None] * factors[:, None]), axis=0)

    return acc


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float = C.G_REAL,
) -> np.ndarray:
    """Return accelerations for each body.

    Parameters
    ----------
    positions : (N, 2) or (N, 3) array
        Current positions in simulation units.
    masses : (N,) array
        Body masses.
    fixed_mask : (N,) boolean array
        True for bodies that should not move.
    g_constant : float, optional
        Gravitational constant to use.

    Returns
    -------
    (N, 2) or (N, 3) ndarray
        Accelerations in metres per second squared matching the input
        dimensionality.
    """

    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)

    n = len(masses)
    if n == 0:
        dim = positions.shape[1] if positions.ndim > 1 else 3
        return np.zeros((0, dim), dtype=np.float64)

    if NUMBA_AVAILABLE:
        return _compute_accelerations_jit(
            positions,
            masses,
            fixed_mask,
            g_constant,
            C.SPACE_SCALE**2,
            C.SOFTENING_FACTOR_SQ,
        )

    return _compute_accelerations_py(positions, masses, fixed_mask, g_constant)


def rk4_step_arrays(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    dt: float,
    g_constant: float = C.G_REAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance bodies using an RK4 step operating on arrays."""

    def deriv(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        return compute_accelerations(pos, masses, fixed_mask, g_constant)

    if positions.shape[1] == 2:
        positions = np.hstack(
            [positions, np.zeros((len(masses), 1), dtype=positions.dtype)]
        )
    if velocities.shape[1] == 2:
        velocities = np.hstack(
            [velocities, np.zeros((len(masses), 1), dtype=velocities.dtype)]
        )

    k1a = deriv(positions, velocities)
    k1v = dt * k1a
    k1p = dt * velocities / C.SPACE_SCALE

    pos_k2 = positions + 0.5 * k1p
    vel_k2 = velocities + 0.5 * k1v
    k2a = deriv(pos_k2, vel_k2)
    k2v = dt * k2a
    k2p = dt * (velocities + 0.5 * k1v) / C.SPACE_SCALE

    pos_k3 = positions + 0.5 * k2p
    vel_k3 = velocities + 0.5 * k2v
    k3a = deriv(pos_k3, vel_k3)
    k3v = dt * k3a
    k3p = dt * (velocities + 0.5 * k2v) / C.SPACE_SCALE

    pos_k4 = positions + k3p
    vel_k4 = velocities + k3v
    k4a = deriv(pos_k4, vel_k4)
    k4v = dt * k4a
    k4p = dt * (velocities + k3v) / C.SPACE_SCALE

    new_pos = positions.copy()
    new_vel = velocities.copy()
    for i in range(len(masses)):
        if fixed_mask[i]:
            continue
        new_pos[i] += (k1p[i] + 2 * k2p[i] + 2 * k3p[i] + k4p[i]) / 6.0
        new_vel[i] += (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6.0

    return new_pos, new_vel


def rk4_step_bodies(bodies, dt: float, g_constant: float = C.G_REAL) -> None:
    """Integrate a list of body objects in place using RK4."""
    positions = np.array([b.pos for b in bodies])
    velocities = np.array([b.vel for b in bodies])
    masses = np.array([b.mass for b in bodies])
    fixed_mask = np.array([getattr(b, "fixed", False) for b in bodies])

    new_pos, new_vel = rk4_step_arrays(
        positions, velocities, masses, fixed_mask, dt, g_constant
    )

    for b, p, v, fixed in zip(bodies, new_pos, new_vel, fixed_mask):
        if not fixed:
            b.pos = p
            b.vel = v
