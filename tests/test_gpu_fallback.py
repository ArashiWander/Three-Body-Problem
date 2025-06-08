import builtins
import numpy as np
import sys

from threebody.integrators import compute_accelerations


def test_gpu_fallback(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ModuleNotFoundError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "cupy", None)

    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    masses = np.array([1.0, 1.0], dtype=float)
    fixed_mask = np.array([False, False])

    acc_gpu = compute_accelerations(pos, masses, fixed_mask, use_gpu=True)
    acc_cpu = compute_accelerations(pos, masses, fixed_mask)

    assert np.allclose(acc_gpu, acc_cpu)
