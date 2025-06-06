import math
from threebody.utils import mass_to_display, distance_to_display, time_to_display
from threebody import constants as C


def test_mass_to_display_ranges():
    assert mass_to_display(0) == "0 kg"
    assert mass_to_display(C.SOLAR_MASS) == "1.00 M\u2609"
    assert mass_to_display(0.5 * C.EARTH_MASS) == "0.50 M\u2295"
    assert mass_to_display(50) == "5.00e+01 kg"


def test_distance_to_display_ranges():
    assert distance_to_display(0) == "0 m"
    assert distance_to_display(C.AU) == "1.00 AU"
    assert distance_to_display(-C.AU) == "-1.00 AU"
    assert distance_to_display(2e6) == "2.00 Mm"
    assert distance_to_display(2000) == "2.00 km"
    assert distance_to_display(50) == "50.0 m"


def test_time_to_display_ranges():
    assert time_to_display(-1) == "N/A"
    assert time_to_display(0) == "0 sec"
    assert time_to_display(2 * 31536000) == "2.0 years"
    assert time_to_display(2 * 86400) == "2.0 days"
    assert time_to_display(7200) == "2.0 hrs"
    assert time_to_display(120) == "2.0 min"
    assert time_to_display(5) == "5.0 sec"
