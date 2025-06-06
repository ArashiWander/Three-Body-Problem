"""Minimal physics utilities for N-body simulations.

This module defines a lightweight :class:`Body` class intended purely for
physics calculations and unit tests.  The pygame-based simulation uses the
more feature rich :class:`~threebody.rendering.Body` instead.
"""
import numpy as np

# Physical constants
G_REAL = 6.67430e-11  # m^3 kg^-1 s^-2
SPACE_SCALE = 5e9     # meters per simulation unit
SOFTENING_FACTOR_SQ = 1.0**2  # m^2 softening


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


def accelerations(bodies, g_constant=G_REAL):
    """Compute accelerations on each body."""
    acc = [np.zeros(2, dtype=float) for _ in bodies]
    for i, bi in enumerate(bodies):
        if bi.fixed:
            continue
        for j, bj in enumerate(bodies):
            if i == j:
                continue
            r_vec = bj.pos - bi.pos
            dist_sq = np.dot(r_vec, r_vec)
            if dist_sq == 0:
                continue
            dist_sq_m = dist_sq * SPACE_SCALE**2
            factor = g_constant * bj.mass / (dist_sq_m + SOFTENING_FACTOR_SQ)
            acc[i] += factor * r_vec / np.sqrt(dist_sq)
    return acc


def perform_rk4_step(bodies, dt, g_constant=G_REAL):
    """Advance bodies using a single RK4 step."""
    pos0 = [b.pos.copy() for b in bodies]
    vel0 = [b.vel.copy() for b in bodies]

    def deriv(pos, vel):
        temp = [Body(b.mass, p, v, b.fixed) for b, p, v in zip(bodies, pos, vel)]
        acc = accelerations(temp, g_constant)
        return acc

    # k1
    a1 = deriv(pos0, vel0)
    k1v = [dt * a for a in a1]
    k1p = [dt * v / SPACE_SCALE for v in vel0]

    # k2
    pos_k2 = [p + 0.5 * k1p[i] for i, p in enumerate(pos0)]
    vel_k2 = [v + 0.5 * k1v[i] for i, v in enumerate(vel0)]
    a2 = deriv(pos_k2, vel_k2)
    k2v = [dt * a for a in a2]
    k2p = [dt * (v + 0.5 * k1v[i]) / SPACE_SCALE for i, v in enumerate(vel0)]

    # k3
    pos_k3 = [p + 0.5 * k2p[i] for i, p in enumerate(pos0)]
    vel_k3 = [v + 0.5 * k2v[i] for i, v in enumerate(vel0)]
    a3 = deriv(pos_k3, vel_k3)
    k3v = [dt * a for a in a3]
    k3p = [dt * (v + 0.5 * k2v[i]) / SPACE_SCALE for i, v in enumerate(vel0)]

    # k4
    pos_k4 = [p + k3p[i] for i, p in enumerate(pos0)]
    vel_k4 = [v + k3v[i] for i, v in enumerate(vel0)]
    a4 = deriv(pos_k4, vel_k4)
    k4v = [dt * a for a in a4]
    k4p = [dt * (v + k3v[i]) / SPACE_SCALE for i, v in enumerate(vel0)]

    for i, b in enumerate(bodies):
        if b.fixed:
            continue
        b.pos += (k1p[i] + 2*k2p[i] + 2*k3p[i] + k4p[i]) / 6.0
        b.vel += (k1v[i] + 2*k2v[i] + 2*k3v[i] + k4v[i]) / 6.0


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
