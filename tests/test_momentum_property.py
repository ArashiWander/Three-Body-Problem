import numpy as np
from hypothesis import given, strategies as st, settings

from threebody.physics import Body
from threebody.physics_utils import step_simulation
import threebody.constants as C


@st.composite
def random_system(draw):
    count = draw(st.integers(min_value=2, max_value=5))
    bodies = []
    for _ in range(count):
        mass = draw(st.floats(1e20, 1e22, allow_nan=False, allow_infinity=False))
        pos = [draw(st.floats(-1e5, 1e5, allow_nan=False, allow_infinity=False)) for _ in range(3)]
        vel = [draw(st.floats(-100, 100, allow_nan=False, allow_infinity=False)) for _ in range(3)]
        bodies.append(Body.from_meters(mass, pos, vel))
    return bodies


@given(random_system())
@settings(max_examples=10)
def test_total_momentum_conserved(bodies):
    total_before = np.sum([b.mass * b.vel for b in bodies], axis=0)
    for _ in range(5):
        step_simulation(bodies, 10.0, C.G_REAL, integrator_type="Symplectic")
    total_after = np.sum([b.mass * b.vel for b in bodies], axis=0)
    assert np.allclose(total_before, total_after, rtol=1e-5, atol=1e-2)
