import numpy as np
from hypothesis import given, strategies as st, settings

from threebody.physics import Body
from threebody.physics_utils import step_simulation
import threebody.constants as C


@st.composite
def random_system(draw):
    count = draw(st.integers(min_value=2, max_value=5))
    bodies = []
    positions_used = []

    for _ in range(count):
        mass = draw(st.floats(1e20, 1e22, allow_nan=False, allow_infinity=False))

        # Ensure bodies don't start too close to each other
        # Use a minimum distance of 100 km to avoid numerical instabilities in testing
        attempts = 0
        while attempts < 100:
            pos = [draw(st.floats(-1e5, 1e5, allow_nan=False,
                                 allow_infinity=False)) for _ in range(3)]

            # Check if this position is far enough from existing bodies
            too_close = False
            for existing_pos in positions_used:
                distance = np.sqrt(sum((p - e)**2 for p, e in zip(pos, existing_pos)))
                if distance < 1e5:  # Minimum distance of 100 km for robust testing
                    too_close = True
                    break

            if not too_close:
                positions_used.append(pos)
                break
            attempts += 1
        else:
            # If we can't find a good position, use a systematic offset
            offset = (len(positions_used) + 1) * 1e5
            pos = [offset, 0, 0]
            positions_used.append(pos)

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
    # Use more realistic tolerances for numerical simulations
    # Allow relative error of 1e-4 (0.01%) and absolute error for small values
    assert np.allclose(total_before, total_after, rtol=1e-4, atol=1e2)
