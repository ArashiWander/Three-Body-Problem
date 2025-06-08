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


def test_gpu_path_uses_xp(monkeypatch):
    import types

    fake_cp = types.SimpleNamespace()
    fake_cp.asarray = np.asarray
    fake_cp.asnumpy = np.asarray
    fake_cp.einsum = np.einsum
    fake_cp.sqrt = np.sqrt
    fake_cp.cross = np.cross
    fake_cp.sum = np.sum
    fake_cp.isfinite = np.isfinite
    fake_cp.errstate = np.errstate
    fake_cp.zeros = np.zeros
    fake_cp.zeros_like = np.zeros_like
    fake_cp.hstack = np.hstack
    fake_cp.inf = np.inf
    fake_cp.float64 = np.float64
    fake_cp.newaxis = np.newaxis

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            return fake_cp
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    def boom(*args, **kwargs):
        raise AssertionError("numpy fallback used")

    monkeypatch.setattr(np, "hstack", boom)
    monkeypatch.setattr(np, "zeros_like", boom)

    pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    masses = np.array([1.0, 1.0], dtype=float)
    fixed_mask = np.array([False, False])

    compute_accelerations(pos, masses, fixed_mask, use_gpu=True)
