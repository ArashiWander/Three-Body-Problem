import time
import numpy as np

from threebody.integrators import compute_accelerations
from threebody.constants import G_REAL, SPACE_SCALE, SOFTENING_FACTOR_SQ


def compute_accelerations_python(positions, masses, fixed_mask, g_constant=G_REAL):
    n = len(masses)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    acc = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        if fixed_mask[i]:
            continue
        r_vec = positions - positions[i]
        dist_sq = np.einsum("ij,ij->i", r_vec, r_vec)
        dist_sq[i] = np.inf
        dist_sq_m = dist_sq * (SPACE_SCALE ** 2)
        inv_dist = np.zeros_like(dist_sq)
        mask = dist_sq > 0.0
        inv_dist[mask] = 1.0 / np.sqrt(dist_sq[mask])
        factors = g_constant * masses / (dist_sq_m + SOFTENING_FACTOR_SQ)
        acc[i] = np.sum(r_vec * (inv_dist[:, None] * factors[:, None]), axis=0)
    return acc


if __name__ == "__main__":
    np.random.seed(0)
    N = 1500  # >1k bodies
    positions = np.random.random((N, 2))
    masses = np.random.random(N) + 1.0
    fixed_mask = np.zeros(N, dtype=bool)

    # warm up JIT
    compute_accelerations(positions, masses, fixed_mask)

    t0 = time.time()
    baseline = compute_accelerations_python(positions, masses, fixed_mask)
    t1 = time.time()
    accelerated = compute_accelerations(positions, masses, fixed_mask)
    t2 = time.time()

    assert np.allclose(baseline, accelerated)
    print(f"Python loop: {t1 - t0:.3f}s")
    print(f"Accelerated : {t2 - t1:.3f}s")
    if t2 - t1 > 0:
        print(f"Speedup     : {(t1 - t0) / (t2 - t1):.1f}x")
