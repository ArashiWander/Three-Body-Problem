from threebody.physics import Body
from threebody.analysis import EnergyMonitor


def test_energy_monitor_basic_update():
    em = EnergyMonitor()
    bodies = [Body(1.0, [0.0, 0.0], [0.0, 0.0]), Body(1.0, [1.0, 0.0], [0.0, 0.0])]
    em.set_initial_energy(bodies, g_constant=1.0)
    em.update(bodies, g_constant=1.0)
    assert len(em.history) == 1
