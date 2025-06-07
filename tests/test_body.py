import numpy as np
from threebody import constants as C
from threebody.rendering import Body
from threebody.physics_utils import detect_and_handle_collisions


def test_set_trail_length_clamps():
    b = Body(1.0, 0, 0, 0, 0, C.WHITE, 5)
    b.set_trail_length(C.MIN_TRAIL_LENGTH - 10)
    assert b.max_trail_length == C.MIN_TRAIL_LENGTH
    b.set_trail_length(C.MAX_TRAIL_LENGTH + 50)
    assert b.max_trail_length == C.MAX_TRAIL_LENGTH


def test_collision_merge_on_true():
    b1 = Body(2 * C.EARTH_MASS, 0, 0, 0, 0, C.WHITE, 5, name="A")
    b2 = Body(C.EARTH_MASS, 0.0005, 0, 0, 0, C.WHITE, 5, name="B")
    bodies = [b1, b2]
    removed = detect_and_handle_collisions(bodies, merge_on_collision=True)
    assert removed == [1]
    assert np.isclose(b1.mass, 3 * C.EARTH_MASS)


def test_collision_bounce_when_merge_false():
    b1 = Body(C.EARTH_MASS, -0.0005, 0, 1.0, 0.0, C.WHITE, 5, name="A")
    b2 = Body(C.EARTH_MASS, 0.0005, 0, -1.0, 0.0, C.WHITE, 5, name="B")
    bodies = [b1, b2]
    removed = detect_and_handle_collisions(bodies, merge_on_collision=False)
    assert removed == []
    assert np.isclose(b1.vel[0], -0.7)
    assert np.isclose(b2.vel[0], 0.7)

def test_collision_bounce_conserves_momentum():
    b1 = Body(C.EARTH_MASS, -0.0005, 0, 1.0, 0.0, C.WHITE, 5, name="A")
    b2 = Body(C.EARTH_MASS, 0.0005, 0, -1.0, 0.0, C.WHITE, 5, name="B")
    bodies = [b1, b2]
    p_before = b1.mass * b1.vel + b2.mass * b2.vel
    detect_and_handle_collisions(bodies, merge_on_collision=False)
    p_after = b1.mass * b1.vel + b2.mass * b2.vel
    assert np.allclose(p_before, p_after)


def test_collision_merge_conserves_momentum():
    b1 = Body(C.EARTH_MASS, -0.0005, 0, 1.0, 0.0, C.WHITE, 5, name="A")
    b2 = Body(C.EARTH_MASS, 0.0005, 0, -1.0, 0.0, C.WHITE, 5, name="B")
    bodies = [b1, b2]
    p_before = b1.mass * b1.vel + b2.mass * b2.vel
    removed = detect_and_handle_collisions(bodies, merge_on_collision=True)
    for idx in removed:
        bodies.pop(idx)
    p_after = bodies[0].mass * bodies[0].vel
    assert np.allclose(p_before, p_after)
    assert np.isclose(bodies[0].mass, 2 * C.EARTH_MASS)

