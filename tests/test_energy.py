import math
from threebody import Body, perform_rk4_step, system_energy, G_REAL, SPACE_SCALE


def test_energy_conservation():
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    # Earth at 1 AU
    r = 1.496e11 / SPACE_SCALE
    v = 29780.0

    sun = Body(sun_mass, [0, 0], [0, 0], fixed=True)
    earth = Body(earth_mass, [r, 0], [0, v])
    bodies = [sun, earth]

    ke0, pe0, e0 = system_energy(bodies, G_REAL)

    dt = 86400  # one day
    for _ in range(30):
        perform_rk4_step(bodies, dt, G_REAL)

    ke1, pe1, e1 = system_energy(bodies, G_REAL)
    assert math.isclose(e0, e1, rel_tol=1e-3)
