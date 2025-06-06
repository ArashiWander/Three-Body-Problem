from threebody.rendering import Body
from threebody.physics_utils import detect_and_handle_collisions
from threebody import constants as C


def test_trail_length_clamping():
    b = Body(1, 0, 0, 0, 0, (255, 255, 255), 5)
    b.set_trail_length(C.MAX_TRAIL_LENGTH + 500)
    assert b.max_trail_length == C.MAX_TRAIL_LENGTH
    b.set_trail_length(C.MIN_TRAIL_LENGTH - 10)
    assert b.max_trail_length == C.MIN_TRAIL_LENGTH


def test_merge_on_collision():
    b1 = Body(C.EARTH_MASS, 0, 0, 0, 0, (255, 255, 255), 5)
    b2 = Body(C.EARTH_MASS, 1e-6, 0.0, 0, 0, (255, 255, 255), 5)
    bodies = [b1, b2]
    indices = detect_and_handle_collisions(bodies, merge_on_collision=True)
    assert indices == [1]
    assert len(bodies) == 2

