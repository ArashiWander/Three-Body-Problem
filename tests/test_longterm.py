import numpy as np
import math
from threebody import Body, perform_rk4_step, system_energy, G_REAL, SPACE_SCALE
from threebody.integrators import compute_accelerations


def _total_momentum(bodies):
    p = np.zeros(3, dtype=float)
    for b in bodies:
        if not getattr(b, "fixed", False):
            p += b.mass * b.vel
    return p


def _leapfrog_step(pos, vel, masses, fixed_mask, dt, g=G_REAL):
    acc = compute_accelerations(pos, masses, fixed_mask, g)
    vel_half = vel + 0.5 * dt * acc
    pos = pos + dt * vel_half / SPACE_SCALE
    acc_new = compute_accelerations(pos, masses, fixed_mask, g)
    vel = vel_half + 0.5 * dt * acc_new
    return pos, vel


def test_energy_momentum_long_term():
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    r = 1.496e11 / SPACE_SCALE
    v = 29780.0

    sun_vel = np.array([0.0, -(earth_mass / sun_mass) * v])
    sun = Body(sun_mass, [0.0, 0.0], sun_vel)
    earth = Body(earth_mass, [r, 0.0], [0.0, v])
    bodies = [sun, earth]

    e0 = system_energy(bodies, G_REAL)[2]
    p0 = _total_momentum(bodies)

    dt = 86400.0
    for _ in range(365 * 5):
        perform_rk4_step(bodies, dt, G_REAL)

    e1 = system_energy(bodies, G_REAL)[2]
    p1 = _total_momentum(bodies)

    assert math.isclose(e0, e1, rel_tol=1e-2)
    assert np.allclose(p0, p1, atol=1e15)


def test_leapfrog_integrator_accuracy():
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    r = 1.496e11 / SPACE_SCALE
    v = 29780.0

    positions = np.array([[0.0, 0.0], [r, 0.0]], dtype=float)
    velocities = np.array(
        [[0.0, -(earth_mass / sun_mass) * v], [0.0, v]], dtype=float
    )
    masses = np.array([sun_mass, earth_mass], dtype=float)
    fixed_mask = np.array([False, False], dtype=bool)

    bodies_start = [Body(masses[i], positions[i], velocities[i]) for i in range(2)]
    e0 = system_energy(bodies_start, G_REAL)[2]

    dt = 86400.0
    for _ in range(365):
        positions, velocities = _leapfrog_step(
            positions, velocities, masses, fixed_mask, dt, G_REAL
        )

    bodies_end = [Body(masses[i], positions[i], velocities[i]) for i in range(2)]
    e1 = system_energy(bodies_end, G_REAL)[2]

    earth_final_pos = positions[1]
    assert math.isclose(e0, e1, rel_tol=1e-6)
    assert np.allclose(earth_final_pos, [r, 0.0], atol=5e-2)

