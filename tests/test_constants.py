import importlib
import threebody.constants as C
from threebody.presets import get_preset_softening_length, PRESET_SOFTENING_LENGTHS


def test_softening_factor_matches_length():
    importlib.reload(C)
    assert C.SOFTENING_FACTOR_SQ == C.SOFTENING_LENGTH**2


def test_get_preset_softening_length():
    importlib.reload(C)
    preset_value = get_preset_softening_length("Sun & Earth", override=0.5)
    assert preset_value == PRESET_SOFTENING_LENGTHS["Sun & Earth"]

    override_value = get_preset_softening_length("Empty", override=0.5)
    assert override_value == 0.5

    default_value = get_preset_softening_length("Empty")
    assert default_value == C.SOFTENING_LENGTH
