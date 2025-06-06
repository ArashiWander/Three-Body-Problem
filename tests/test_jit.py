import numpy as np

from threebody import constants as C
from threebody.integrators import compute_accelerations
from threebody.jit import calculate_acceleration_jit


def test_jit_matches_python_multiple_bodies():
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float
    )
    masses = np.array([1.0, 2.0, 3.0], dtype=float)
    fixed = np.array([False, True, False])

    acc_python = compute_accelerations(positions, masses, fixed, g_constant=1.0)

    acc_jit = np.zeros_like(acc_python)
    for i in range(len(masses)):
        if fixed[i]:
            continue
        other_pos = np.concatenate([positions[:i], positions[i + 1 :]])
        other_mass = np.concatenate([masses[:i], masses[i + 1 :]])
        acc_jit[i] = calculate_acceleration_jit(
            positions[i],
            masses[i],
            other_pos,
            other_mass,
            1.0,
            C.SOFTENING_FACTOR_SQ,
            C.SPACE_SCALE,
        )

    assert np.allclose(acc_jit, acc_python)

