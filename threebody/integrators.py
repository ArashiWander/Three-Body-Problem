from __future__ import annotations

import numpy as np

from . import constants as C

try:  # optional GPU acceleration
    import cupy as cp  # type: ignore

    try:
        cp.cuda.runtime.getDeviceCount()
        _CUPY_GPU_AVAILABLE = True
    except Exception:  # pragma: no cover - no GPU available
        _CUPY_GPU_AVAILABLE = False
except Exception:  # pragma: no cover - CuPy not installed
    cp = None  # type: ignore
    _CUPY_GPU_AVAILABLE = False


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
    ndarray
        Accelerations in metres per second squared with the same dimensionality
        as ``positions``.

    Notes
    -----
    If CuPy is installed and a CUDA-capable GPU is detected, the computation
    runs on the GPU automatically. If GPU acceleration is unavailable or any
    error occurs, the function falls back to the CPU implementation.
    """

    if _CUPY_GPU_AVAILABLE:
        try:  # pragma: no cover - GPU path not tested without hardware
            return _compute_accelerations_gpu(
                positions, masses, fixed_mask, g_constant
            )
        except Exception:
            pass

    return _compute_accelerations_cpu(
        positions, masses, fixed_mask, g_constant
    )


def _compute_accelerations_cpu(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float,
) -> np.ndarray:
    """Compute accelerations on the CPU using symmetric pair updates."""

    n = len(masses)
    if n == 0:
        return np.zeros((0, positions.shape[1]), dtype=np.float64)

    orig_dim = positions.shape[1]
    if orig_dim == 2:
        positions = np.hstack([
            positions,
            np.zeros((n, 1), dtype=positions.dtype),
        ])
        dim = 3
    else:
        dim = orig_dim

    acc = np.zeros((n, dim), dtype=np.float64)
    scale_sq = C.SPACE_SCALE ** 2

    for i in range(n - 1):
        for j in range(i + 1, n):
            r_vec = positions[j] - positions[i]
            dist_sq = float(np.dot(r_vec, r_vec))
            if dist_sq == 0.0:
                continue
            dist_sq_m = dist_sq * scale_sq
            inv_dist = 1.0 / np.sqrt(dist_sq)
            factor = g_constant / (dist_sq_m + C.SOFTENING_FACTOR_SQ)
            acc_vec = r_vec * (inv_dist * factor)

            if not fixed_mask[i]:
                acc[i] += acc_vec * masses[j]
            if not fixed_mask[j]:
                acc[j] -= acc_vec * masses[i]

    return acc[:, :orig_dim]


def _compute_accelerations_gpu(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float,
) -> np.ndarray:
    pos_gpu = cp.asarray(positions, dtype=cp.float64)
    masses_gpu = cp.asarray(masses, dtype=cp.float64)

    n = len(masses)
    if n == 0:
        return np.zeros((0, positions.shape[1]), dtype=np.float64)

    orig_dim = positions.shape[1]
    if orig_dim == 2:
        pos_gpu = cp.hstack([pos_gpu, cp.zeros((n, 1), dtype=pos_gpu.dtype)])
        dim = 3
    else:
        dim = orig_dim

    acc_gpu = cp.zeros((n, dim), dtype=cp.float64)
    for i in range(n):
        if fixed_mask[i]:
            continue

        r_vec = pos_gpu - pos_gpu[i]
        dist_sq = cp.einsum("ij,ij->i", r_vec, r_vec)
        dist_sq[i] = cp.inf

        dist_sq_m = dist_sq * (C.SPACE_SCALE ** 2)

        inv_dist = cp.where(dist_sq > 0.0, 1.0 / cp.sqrt(dist_sq), 0.0)

        factors = g_constant * masses_gpu / (dist_sq_m + C.SOFTENING_FACTOR_SQ)

        acc_gpu[i] = cp.sum(r_vec * (inv_dist[:, None] * factors[:, None]), axis=0)

    return cp.asnumpy(acc_gpu)[:, :orig_dim]


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

    orig_pos_dim = positions.shape[1]
    orig_vel_dim = velocities.shape[1]

    if orig_pos_dim == 2:
        positions = np.hstack([
            positions,
            np.zeros((len(masses), 1), dtype=positions.dtype),
        ])
    if orig_vel_dim == 2:
        velocities = np.hstack([
            velocities,
            np.zeros((len(masses), 1), dtype=velocities.dtype),
        ])

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

    return new_pos[:, :orig_pos_dim], new_vel[:, :orig_vel_dim]


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
