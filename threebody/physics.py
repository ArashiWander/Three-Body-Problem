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
    def __init__(self, mass, pos, vel, fixed=False):
        self.mass = float(mass)
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
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
