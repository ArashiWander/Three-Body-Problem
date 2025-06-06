import pytest
from threebody import constants as C
from threebody.rendering import Body


def test_init_clamps_trail_length_low():
    b = Body(1.0, 0, 0, 0, 0, (255,255,255), 5, max_trail_length=C.MIN_TRAIL_LENGTH-10)
    assert b.max_trail_length == C.MIN_TRAIL_LENGTH
    assert b.trail.maxlen == C.MIN_TRAIL_LENGTH


def test_init_clamps_trail_length_high():
    b = Body(1.0, 0, 0, 0, 0, (255,255,255), 5, max_trail_length=C.MAX_TRAIL_LENGTH+10)
    assert b.max_trail_length == C.MAX_TRAIL_LENGTH
    assert b.trail.maxlen == C.MAX_TRAIL_LENGTH


def test_set_trail_length_clamps_low():
    b = Body(1.0, 0, 0, 0, 0, (255,255,255), 5)
    b.set_trail_length(C.MIN_TRAIL_LENGTH-20)
    assert b.max_trail_length == C.MIN_TRAIL_LENGTH
    assert b.trail.maxlen == C.MIN_TRAIL_LENGTH


def test_set_trail_length_clamps_high():
    b = Body(1.0, 0, 0, 0, 0, (255,255,255), 5)
    b.set_trail_length(C.MAX_TRAIL_LENGTH+20)
    assert b.max_trail_length == C.MAX_TRAIL_LENGTH
    assert b.trail.maxlen == C.MAX_TRAIL_LENGTH
