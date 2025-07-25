import numpy as np
from hypothesis import given, strategies as st, settings

from threebody.physics import Body
from threebody.physics_utils import step_simulation
import threebody.constants as C


def test_total_momentum_conserved_simple_cases():
    """Test momentum conservation with hand-crafted, physically reasonable scenarios."""
    
    # Test Case 1: Two equal masses with zero initial velocity, separated by reasonable distance
    bodies = [
        Body.from_meters(1e24, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        Body.from_meters(1e24, [10000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    ]
    
    total_before = np.sum([b.mass * b.vel for b in bodies], axis=0)
    for _ in range(10):
        step_simulation(bodies, 10.0, C.G_REAL, integrator_type="Symplectic")
    total_after = np.sum([b.mass * b.vel for b in bodies], axis=0)
    
    assert np.allclose(total_before, total_after, rtol=1e-10, atol=1e-5), \
        f"Case 1 failed: {total_before} vs {total_after}"
    
    # Test Case 2: Three-body system with reasonable separations
    bodies = [
        Body.from_meters(2e24, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        Body.from_meters(1e24, [20000.0, 0.0, 0.0], [0.0, 50.0, 0.0]),
        Body.from_meters(1e24, [0.0, 20000.0, 0.0], [-50.0, 0.0, 0.0])
    ]
    
    total_before = np.sum([b.mass * b.vel for b in bodies], axis=0)
    for _ in range(5):
        step_simulation(bodies, 10.0, C.G_REAL, integrator_type="Symplectic")
    total_after = np.sum([b.mass * b.vel for b in bodies], axis=0)
    
    assert np.allclose(total_before, total_after, rtol=1e-8, atol=1e-3), \
        f"Case 2 failed: {total_before} vs {total_after}"
    
    # Test Case 3: Earth-Sun like system
    sun_mass = 1.989e30
    earth_mass = 5.972e24
    earth_distance = 1.496e11  # 1 AU
    earth_velocity = 29780  # m/s
    
    bodies = [
        Body.from_meters(sun_mass, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        Body.from_meters(earth_mass, [earth_distance, 0.0, 0.0], [0.0, earth_velocity, 0.0])
    ]
    
    total_before = np.sum([b.mass * b.vel for b in bodies], axis=0)
    # Shorter simulation time for stability
    for _ in range(3):
        step_simulation(bodies, 86400.0, C.G_REAL, integrator_type="Symplectic")  # 1 day steps
    total_after = np.sum([b.mass * b.vel for b in bodies], axis=0)
    
    assert np.allclose(total_before, total_after, rtol=1e-6, atol=1e5), \
        f"Case 3 failed: {total_before} vs {total_after}"


@st.composite
def reasonable_system(draw):
    """Generate physically reasonable systems for testing."""
    count = draw(st.integers(min_value=2, max_value=3))  # Limit to 2-3 bodies for stability
    bodies = []
    
    for i in range(count):
        mass = draw(st.floats(1e22, 1e25, allow_nan=False, allow_infinity=False))
        
        # Ensure bodies are well-separated (minimum 1000 km)
        if i == 0:
            pos = [0.0, 0.0, 0.0]
        else:
            # Place on a circle with radius 10-100 km
            angle = (2 * np.pi * i) / count
            radius = draw(st.floats(50000.0, 500000.0))  # 50-500 km
            pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
        
        # Reasonable velocities for space objects
        vel = [draw(st.floats(-1000, 1000)) for _ in range(3)]  # km/s range
        bodies.append(Body.from_meters(mass, pos, vel))
    
    return bodies


@given(reasonable_system())
@settings(max_examples=5)
def test_momentum_conservation_property(bodies):
    """Property-based test with reasonable constraints."""
    total_before = np.sum([b.mass * b.vel for b in bodies], axis=0)
    
    # Short simulation with small steps for numerical stability
    for _ in range(2):
        step_simulation(bodies, 1.0, C.G_REAL, integrator_type="Symplectic")
    
    total_after = np.sum([b.mass * b.vel for b in bodies], axis=0)
    
    # Allow for small numerical errors but momentum should be largely preserved
    momentum_magnitude = np.linalg.norm(total_before)
    if momentum_magnitude > 0:
        relative_error = np.linalg.norm(total_after - total_before) / momentum_magnitude
        assert relative_error < 0.01, f"Large relative momentum error: {relative_error}"
    else:
        # If initial momentum is zero, final should be small
        assert np.linalg.norm(total_after) < 1e10, f"Zero initial momentum but large final: {total_after}"
