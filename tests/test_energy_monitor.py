import threebody.constants as C
from threebody.physics import Body
from threebody.analysis import EnergyMonitor


def test_energy_monitor_basic_update():
    em = EnergyMonitor()
    bodies = [Body(1.0, [0.0, 0.0], [0.0, 0.0]), Body(1.0, [1.0, 0.0], [0.0, 0.0])]
    em.set_initial_energy(bodies, g_constant=1.0)
    em.update(bodies, g_constant=1.0)
    assert len(em.history) == 1


def test_energy_monitor_export_csv(tmp_path):
    em = EnergyMonitor()
    bodies = [Body(1.0, [0.0, 0.0], [0.0, 0.0]), Body(1.0, [1.0, 0.0], [0.0, 0.0])]
    em.set_initial_energy(bodies, g_constant=1.0)
    for _ in range(3):
        em.update(bodies, g_constant=1.0)

    out_file = tmp_path / "hist.csv"
    em.export_csv(out_file)
    lines = out_file.read_text().strip().splitlines()
    assert lines[0] == "step,energy_drift_percent"
    assert len(lines) == 4
