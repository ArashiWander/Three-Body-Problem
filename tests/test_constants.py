import importlib
import threebody.constants as C


def test_softening_factor_matches_length():
    importlib.reload(C)
    assert C.SOFTENING_FACTOR_SQ == C.SOFTENING_LENGTH**2


def test_speed_of_light_constant():
    importlib.reload(C)
    assert C.C_LIGHT == 299792458
