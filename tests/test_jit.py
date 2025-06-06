import numpy as np
from threebody.jit import apply_boundary_conditions_jit


def test_apply_boundary_conditions_jit_reflects():
    pos = np.array([1.2, 0.5])
    vel = np.array([1.0, 0.0])
    bounds = (0.0, 0.0, 1.0, 1.0)
    new_pos, new_vel = apply_boundary_conditions_jit(pos, vel, bounds, 0.5)
    assert np.allclose(new_pos, [0.9, 0.5])
    assert np.allclose(new_vel, [-0.5, 0.0])


def test_apply_boundary_conditions_jit_no_change():
    pos = np.array([0.5, 0.5])
    vel = np.array([0.1, -0.2])
    bounds = (0.0, 0.0, 1.0, 1.0)
    new_pos, new_vel = apply_boundary_conditions_jit(pos, vel, bounds, 0.8)
    assert np.allclose(new_pos, pos)
    assert np.allclose(new_vel, vel)
