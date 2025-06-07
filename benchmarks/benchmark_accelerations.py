import os
import time
import importlib
import numpy as np

N_BODIES = 1200

positions = np.random.rand(N_BODIES, 3)
masses = np.random.rand(N_BODIES)
fixed_mask = np.zeros(N_BODIES, dtype=bool)


def run(use_numba: bool) -> float:
    if use_numba:
        os.environ.pop("NUMBA_DISABLE_JIT", None)
    else:
        os.environ["NUMBA_DISABLE_JIT"] = "1"
    import threebody.integrators as integrators

    importlib.reload(integrators)
    start = time.perf_counter()
    integrators.compute_accelerations(positions, masses, fixed_mask)
    return time.perf_counter() - start


if __name__ == "__main__":
    t_numba = run(True)
    t_py = run(False)
    print(f"Numba enabled: {t_numba:.4f}s")
    print(f"Python only : {t_py:.4f}s")
    if t_py > 0:
        print(f"Speedup     : {t_py / t_numba:.2f}x")
