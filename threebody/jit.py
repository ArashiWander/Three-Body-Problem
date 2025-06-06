"""JIT accelerated helpers using numba when available.

If numba fails to import or initialize for any reason, these helpers fall back
to pure Python implementations so the package continues to work without JIT
acceleration.
"""

import numpy as np

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba optional or misconfigured
    def nb_njit(func):
        return func

    class DummyNB:
        njit = staticmethod(nb_njit)

    nb = DummyNB()
    NUMBA_AVAILABLE = False


@nb.njit
def calculate_acceleration_jit(target_pos_sim, target_mass_kg, other_positions_sim,
                               other_masses_kg, g_const, softening_sq_m2, space_scale):
    acc = np.zeros(2, dtype=np.float64)
    num_others = len(other_masses_kg)
    for i in range(num_others):
        distance_vec_sim = other_positions_sim[i] - target_pos_sim
        dist_sq_sim = distance_vec_sim[0] ** 2 + distance_vec_sim[1] ** 2
        if dist_sq_sim == 0:
            continue
        dist_sq_meters = dist_sq_sim * (space_scale ** 2)
        acc_mag = g_const * other_masses_kg[i] / (dist_sq_meters + softening_sq_m2)
        dist_sim = np.sqrt(dist_sq_sim)
        direction_vec = distance_vec_sim / dist_sim
        acc += direction_vec * acc_mag
    return acc


@nb.njit
def apply_boundary_conditions_jit(pos_sim, vel_m_s, bounds_sim, elasticity):
    min_x, min_y, max_x, max_y = bounds_sim
    pos_out = pos_sim.copy()
    vel_out = vel_m_s.copy()
    collided = False
    if pos_out[0] < min_x:
        pos_out[0] = min_x + (min_x - pos_out[0]) * elasticity
        vel_out[0] *= -elasticity
        collided = True
    elif pos_out[0] > max_x:
        pos_out[0] = max_x - (pos_out[0] - max_x) * elasticity
        vel_out[0] *= -elasticity
        collided = True
    if pos_out[1] < min_y:
        pos_out[1] = min_y + (min_y - pos_out[1]) * elasticity
        vel_out[1] *= -elasticity
        collided = True
    elif pos_out[1] > max_y:
        pos_out[1] = max_y - (pos_out[1] - max_y) * elasticity
        vel_out[1] *= -elasticity
        collided = True
    if collided:
        pos_out[0] = max(min_x, min(pos_out[0], max_x))
        pos_out[1] = max(min_y, min(pos_out[1], max_y))
    return pos_out, vel_out
