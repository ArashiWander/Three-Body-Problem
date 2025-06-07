import numpy as np
import math

from threebody.physics import Body, accelerations, perform_rk4_step, SPACE_SCALE
from threebody.physics_utils import adaptive_rk4_step


def test_accelerations_fixed_body():
    b_fixed = Body(1.0, [0.0, 0.0], [0.0, 0.0], fixed=True)
    b_free = Body(1.0, [1.0, 0.0], [0.0, 0.0])
    acc = accelerations([b_fixed, b_free], g_constant=1.0)
    # fixed body should have zero acceleration
    assert np.allclose(acc[0], [0.0, 0.0, 0.0])
    # expected acceleration on free body towards the fixed body
    expected = -1.0 / (SPACE_SCALE ** 2 + 1.0)
    assert math.isclose(acc[1][0], expected, rel_tol=1e-12)
    assert math.isclose(acc[1][1], 0.0, abs_tol=1e-12)
    assert math.isclose(acc[1][2], 0.0, abs_tol=1e-12)


def test_accelerations_zero_distance():
    b1 = Body(1.0, [0.0, 0.0], [0.0, 0.0])
    b2 = Body(2.0, [0.0, 0.0], [0.0, 0.0])
    acc = accelerations([b1, b2], g_constant=1.0)
    # zero distance should result in zero acceleration due to skip
    assert np.allclose(acc[0], [0.0, 0.0, 0.0])
    assert np.allclose(acc[1], [0.0, 0.0, 0.0])


def test_rk4_step_keeps_fixed_body_static():
    fixed = Body(1.0, [0.0, 0.0], [0.0, 0.0], fixed=True)
    mover = Body(1.0, [1.0, 0.0], [0.0, 1.0])
    bodies = [fixed, mover]
    perform_rk4_step(bodies, dt=1.0, g_constant=0.0)
    assert np.allclose(fixed.pos, [0.0, 0.0, 0.0])
    assert np.allclose(fixed.vel, [0.0, 0.0, 0.0])


def test_rk4_step_zero_distance():
    b1 = Body(1.0, [0.0, 0.0], [1.0, 0.0])
    b2 = Body(1.0, [0.0, 0.0], [-1.0, 0.0])
    bodies = [b1, b2]
    perform_rk4_step(bodies, dt=1.0, g_constant=0.0)
    # with no gravity, motion should be purely linear despite zero distance
    expected_disp = 1.0 / SPACE_SCALE
    assert math.isclose(b1.pos[0], expected_disp, rel_tol=1e-12)
    assert math.isclose(b2.pos[0], -expected_disp, rel_tol=1e-12)


def test_adaptive_rk4_step_basic():
    body = Body(1.0, [0.0, 0.0], [1.0, 0.0])
    dt, dt_new = adaptive_rk4_step(
        [body],
        20.0,
        g_constant=0.0,
        error_tolerance=1e-6,
        use_boundaries=False,
        bounds_sim=None,
    )
    assert dt == 20.0
    assert dt_new == 40.0
    expected = 20.0 / SPACE_SCALE
    assert math.isclose(body.pos[0], expected, rel_tol=1e-12)
    assert math.isclose(body.vel[0], 1.0, rel_tol=1e-12)
