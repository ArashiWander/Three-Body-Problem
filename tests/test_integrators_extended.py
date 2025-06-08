import numpy as np
import math

from threebody import Body, perform_rk4_step, system_energy, G_REAL, SPACE_SCALE
from threebody.integrators import compute_accelerations
from threebody.physics_utils import detect_and_handle_collisions
from threebody.integrators import symplectic4_step_arrays


def _total_momentum(bodies):
    p = np.zeros(3, dtype=float)
    for b in bodies:
        if not getattr(b, "fixed", False):
            p += b.mass * b.vel
    return p


def _symplectic_step_bodies(bodies, dt, g_constant=G_REAL):
    pos = np.array([b.pos for b in bodies], dtype=float)
    vel = np.array([b.vel for b in bodies], dtype=float)
    masses = np.array([b.mass for b in bodies], dtype=float)
    fixed_mask = np.array([b.fixed for b in bodies], dtype=bool)

    acc = compute_accelerations(pos, masses, fixed_mask, g_constant)
    vel_half = vel + 0.5 * dt * acc
    pos = pos + dt * vel_half / SPACE_SCALE
    acc_new = compute_accelerations(pos, masses, fixed_mask, g_constant)
    vel = vel_half + 0.5 * dt * acc_new

    for b, p, v, fixed in zip(bodies, pos, vel, fixed_mask):
        if not fixed:
            b.pos = p
            b.vel = v


def _symplectic4_step_bodies(bodies, dt, g_constant=G_REAL):
    pos = np.array([b.pos for b in bodies], dtype=float)
    vel = np.array([b.vel for b in bodies], dtype=float)
    masses = np.array([b.mass for b in bodies], dtype=float)
    fixed_mask = np.array([b.fixed for b in bodies], dtype=bool)

    pos, vel = symplectic4_step_arrays(
        pos, vel, masses, fixed_mask, dt, g_constant
    )

    for b, p, v, fixed in zip(bodies, pos, vel, fixed_mask):
        if not fixed:
            b.pos = p
            b.vel = v


def _init_orbit_bodies():
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    r = 1.496e11 / SPACE_SCALE
    v = 29780.0
    sun_vel = np.array([0.0, -(earth_mass / sun_mass) * v])
    sun = Body(sun_mass, [0.0, 0.0], sun_vel)
    earth = Body(earth_mass, [r, 0.0], [0.0, v])
    return [sun, earth]


def test_multi_year_rk4_stability():
    bodies = _init_orbit_bodies()
    e0 = system_energy(bodies, G_REAL)[2]
    p0 = _total_momentum(bodies)

    dt = 86400.0
    for _ in range(365 * 10):
        perform_rk4_step(bodies, dt, G_REAL)

    e1 = system_energy(bodies, G_REAL)[2]
    p1 = _total_momentum(bodies)

    assert math.isclose(e0, e1, rel_tol=2e-2)
    assert np.allclose(p0, p1, atol=1e15)


def test_multi_year_symplectic_stability():
    bodies = _init_orbit_bodies()
    e0 = system_energy(bodies, G_REAL)[2]
    p0 = _total_momentum(bodies)

    dt = 86400.0
    for _ in range(365 * 10):
        _symplectic_step_bodies(bodies, dt, G_REAL)

    e1 = system_energy(bodies, G_REAL)[2]
    p1 = _total_momentum(bodies)

    assert math.isclose(e0, e1, rel_tol=1e-5)
    assert np.allclose(p0, p1, atol=1e15)


def test_multi_year_symplectic4_stability():
    bodies = _init_orbit_bodies()
    e0 = system_energy(bodies, G_REAL)[2]
    p0 = _total_momentum(bodies)

    dt = 86400.0
    for _ in range(365 * 10):
        _symplectic4_step_bodies(bodies, dt, G_REAL)

    e1 = system_energy(bodies, G_REAL)[2]
    p1 = _total_momentum(bodies)

    assert math.isclose(e0, e1, rel_tol=1e-6)
    assert np.allclose(p0, p1, atol=1e15)


def _init_collision_bodies():
    from threebody import constants as C
    b1 = Body(C.EARTH_MASS, [-0.007, 0.0], [1000.0, 0.0])
    b2 = Body(C.EARTH_MASS, [0.007, 0.0], [-1000.0, 0.0])
    for name, b in zip(["A", "B"], [b1, b2]):
        b.radius_pixels = 5
        b.name = name
        b.clear_trail = lambda: None
    return [b1, b2]


def _run_collision_sim(bodies, step_func):
    dt = 1000.0
    p0 = _total_momentum(bodies)
    for _ in range(6):
        step_func(bodies, dt, g_constant=0.0)
        detect_and_handle_collisions(bodies, merge_on_collision=False)
    p1 = _total_momentum(bodies)
    return p0, p1, bodies


def test_rk4_collision_bounce():
    bodies = _init_collision_bodies()
    p0, p1, bodies = _run_collision_sim(bodies, perform_rk4_step)
    b1, b2 = bodies
    assert b1.vel[0] < 0 and b2.vel[0] > 0
    assert np.allclose(p0, p1)


def test_symplectic_collision_bounce():
    bodies = _init_collision_bodies()
    p0, p1, bodies = _run_collision_sim(bodies, _symplectic_step_bodies)
    b1, b2 = bodies
    assert b1.vel[0] < 0 and b2.vel[0] > 0
    assert np.allclose(p0, p1)


def test_symplectic4_collision_bounce():
    bodies = _init_collision_bodies()
    p0, p1, bodies = _run_collision_sim(bodies, _symplectic4_step_bodies)
    b1, b2 = bodies
    assert b1.vel[0] < 0 and b2.vel[0] > 0
    assert np.allclose(p0, p1)
