from __future__ import annotations

import numpy as np

from . import constants as C


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float = C.G_REAL,
) -> np.ndarray:
    """Return accelerations for each body.

    Parameters
    ----------
    positions : (N, 3) array
        Current positions in simulation units. 2-D arrays are also accepted and
        treated as lying in the ``z=0`` plane.
    masses : (N,) array
        Body masses.
    fixed_mask : (N,) boolean array
        True for bodies that should not move.
    g_constant : float, optional
        Gravitational constant to use.

    Returns
    -------
    (N, 3) ndarray
        Accelerations in metres per second squared.
    """

    n = len(masses)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    original_dim = positions.shape[1]
    if original_dim == 2:
        positions_3d = np.hstack([positions, np.zeros((n, 1), dtype=positions.dtype)])
    else:
        positions_3d = positions

    acc = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if fixed_mask[i]:
            continue

        r_vec = positions_3d - positions_3d[i]
        dist_sq = np.einsum("ij,ij->i", r_vec, r_vec)
        dist_sq[i] = np.inf  # ignore self

        dist_sq_m = dist_sq * (C.SPACE_SCALE**2)

        inv_dist = np.zeros_like(dist_sq)
        mask = dist_sq > 0.0
        inv_dist[mask] = 1.0 / np.sqrt(dist_sq[mask])

        factors = g_constant * masses / (dist_sq_m + C.SOFTENING_FACTOR_SQ)

        acc[i] = np.sum(r_vec * (inv_dist[:, None] * factors[:, None]), axis=0)

    if original_dim == 2:
        return acc[:, :2]
    return acc


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
        positions = np.hstack([positions, np.zeros((len(masses), 1), dtype=positions.dtype)])
    if velocities.shape[1] == 2:
        velocities = np.hstack([velocities, np.zeros((len(masses), 1), dtype=velocities.dtype)])

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


def symplectic_step_arrays(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    dt: float,
    g_constant: float = C.G_REAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance bodies using a velocity Verlet (symplectic) step."""

    if positions.shape[1] == 2:
        positions = np.hstack([positions, np.zeros((len(masses), 1), dtype=positions.dtype)])
    if velocities.shape[1] == 2:
        velocities = np.hstack([velocities, np.zeros((len(masses), 1), dtype=velocities.dtype)])

    acc0 = compute_accelerations(positions, masses, fixed_mask, g_constant)

    half_vel = velocities + 0.5 * dt * acc0
    new_pos = positions + dt * half_vel / C.SPACE_SCALE

    acc1 = compute_accelerations(new_pos, masses, fixed_mask, g_constant)
    new_vel = half_vel + 0.5 * dt * acc1

    for i in range(len(masses)):
        if fixed_mask[i]:
            new_pos[i] = positions[i]
            new_vel[i] = velocities[i]

    return new_pos, new_vel


def symplectic_step_bodies(bodies, dt: float, g_constant: float = C.G_REAL) -> None:
    """Integrate a list of body objects in place using velocity Verlet."""
    positions = np.array([b.pos for b in bodies])
    velocities = np.array([b.vel for b in bodies])
    masses = np.array([b.mass for b in bodies])
    fixed_mask = np.array([getattr(b, "fixed", False) for b in bodies])

    new_pos, new_vel = symplectic_step_arrays(
        positions, velocities, masses, fixed_mask, dt, g_constant
    )

    for b, p, v, fixed in zip(bodies, new_pos, new_vel, fixed_mask):
        if not fixed:
            b.pos = p
            b.vel = v
