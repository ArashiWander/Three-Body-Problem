"""Minimal physics utilities for N-body simulations.

This module defines a lightweight :class:`Body` class intended purely for
physics calculations and unit tests.  The pygame-based simulation uses the
more feature rich :class:`~threebody.rendering.Body` instead.
"""
import numpy as np

from .constants import G_REAL, SPACE_SCALE
from .integrators import compute_accelerations, rk4_step_arrays


class Body:
    """Simple body representation for physics computations."""

    def __init__(self, mass, pos, vel, fixed: bool = False):
        """Create a body.

        Parameters
        ----------
        mass : float
            Mass of the body in kilograms.
        pos : array-like
            Initial position. 2-D values are padded with ``z=0``.
        vel : array-like
            Initial velocity. 2-D values are padded with ``z=0``.
        fixed : bool, optional
            If True the body does not move when integrated.
        """

        self.mass = float(mass)
        p = np.asarray(pos, dtype=float)
        if p.size == 2:
            p = np.append(p, 0.0)
        self.pos = p.astype(float)

        v = np.asarray(vel, dtype=float)
        if v.size == 2:
            v = np.append(v, 0.0)
        self.vel = v.astype(float)

        self.fixed = fixed

    def __repr__(self):
        return (
            f"Body(mass={self.mass}, pos={self.pos.tolist()}, "
            f"vel={self.vel.tolist()}, fixed={self.fixed})"
        )

    @staticmethod
    def from_meters(mass, pos_m, vel_m_s, fixed=False):
        """Create a :class:`Body` using metre based coordinates."""
        pos_sim = np.asarray(pos_m, dtype=float) / SPACE_SCALE
        return Body(mass, pos_sim, vel_m_s, fixed=fixed)


def accelerations(bodies, g_constant=G_REAL):
    """Compute accelerations on each body."""
    if not bodies:
        return []

    positions = np.array([b.pos for b in bodies], dtype=float)
    masses = np.array([b.mass for b in bodies], dtype=float)
    fixed_mask = np.array([b.fixed for b in bodies], dtype=bool)

    acc_array = compute_accelerations(positions, masses, fixed_mask, g_constant)
    return [acc_array[i] for i in range(len(bodies))]


def perform_rk4_step(bodies, dt, g_constant=G_REAL):
    """Advance bodies using a single RK4 step."""
    if not bodies:
        return

    positions = np.array([b.pos for b in bodies], dtype=float)
    velocities = np.array([b.vel for b in bodies], dtype=float)
    masses = np.array([b.mass for b in bodies], dtype=float)
    fixed_mask = np.array([b.fixed for b in bodies], dtype=bool)

    new_pos, new_vel = rk4_step_arrays(
        positions, velocities, masses, fixed_mask, dt, g_constant
    )

    for b, p, v, fixed in zip(bodies, new_pos, new_vel, fixed_mask):
        if not fixed:
            b.pos = p
            b.vel = v


def system_energy(bodies, g_constant=G_REAL):
    """Return total kinetic and potential energy."""
    kinetic = 0.0
    potential = 0.0
    for b in bodies:
        if b.fixed:
            continue
        kinetic += 0.5 * b.mass * np.dot(b.vel, b.vel)
    for i, bi in enumerate(bodies):
        for j, bj in enumerate(bodies[i+1:], i+1):
            r = np.linalg.norm(bj.pos - bi.pos) * SPACE_SCALE
            if r == 0:
                continue
            potential -= g_constant * bi.mass * bj.mass / r
    return kinetic, potential, kinetic + potential
