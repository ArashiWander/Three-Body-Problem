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
