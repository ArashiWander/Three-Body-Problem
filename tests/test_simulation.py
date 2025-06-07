import os
from threebody.simulation import Simulation
from threebody.presets import PRESETS


def test_load_preset_creates_bodies():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sim = Simulation(init_pygame=False)
    sim.load_preset("Sun & Earth")
    assert len(sim.bodies) == len(PRESETS["Sun & Earth"])
