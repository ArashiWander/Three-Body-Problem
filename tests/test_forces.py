import math
import numpy as np

from threebody.physics import Body, forces
from threebody.constants import SPACE_SCALE, SOFTENING_FACTOR_SQ


def test_two_body_forces():
    m1 = 1.0
    m2 = 2.0
    r_m = 1000.0
    r_sim = r_m / SPACE_SCALE
    b1 = Body(m1, [0.0, 0.0], [0.0, 0.0])
    b2 = Body(m2, [r_sim, 0.0], [0.0, 0.0])

    f = forces([b1, b2], g_constant=1.0)

    expected_mag = m1 * m2 / (r_m ** 2 + SOFTENING_FACTOR_SQ)
    assert math.isclose(f[0][0], expected_mag, rel_tol=1e-12)
    assert math.isclose(f[1][0], -expected_mag, rel_tol=1e-12)
    # y and z components should be zero
    assert np.allclose(f[0][1:], [0.0, 0.0])
    assert np.allclose(f[1][1:], [0.0, 0.0])
