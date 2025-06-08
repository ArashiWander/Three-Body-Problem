import numpy as np
from . import constants as C

def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    fixed_mask: np.ndarray,
    g_constant: float = C.G_REAL,
    use_gr_correction: bool = False,
    velocities: np.ndarray = None,  # GR修正需要速度
    use_gpu: bool = False,
) -> np.ndarray:
    """计算每个天体的加速度，可选广义相对论修正。

    Parameters
    ----------
    use_gpu : bool, optional
        If True and the ``cupy`` package is available, calculations are
        performed on the GPU. When ``cupy`` is missing or initialization fails
        the function automatically falls back to NumPy on the CPU.
    """
    xp = np
    if use_gpu:
        try:
            import cupy as cp
        except Exception:
            use_gpu = False
        else:
            xp = cp
            positions = cp.asarray(positions)
            masses = cp.asarray(masses)
            fixed_mask = cp.asarray(fixed_mask)
            if velocities is not None:
                velocities = cp.asarray(velocities)

    n = len(masses)
    if n == 0:
        # 确保返回与输入形状匹配的空数组
        return np.zeros_like(positions, dtype=np.float64)

    # 确保位置和速度是3D的
    if positions.shape[1] == 2:
        positions_3d = np.hstack([positions, np.zeros((n, 1), dtype=positions.dtype)])
    else:
        positions_3d = positions
    
    if velocities is not None and velocities.shape[1] == 2:
        velocities_3d = np.hstack([velocities, np.zeros((n, 1), dtype=velocities.dtype)])
    else:
        velocities_3d = velocities

    acc = np.zeros_like(positions_3d, dtype=np.float64)
    scale_sq = C.SPACE_SCALE ** 2

    for i in range(n):
        if fixed_mask[i]:
            continue
            
        # 为了提高效率，只计算对天体i的作用力
        # 使用向量化计算与天体i的相对位置和距离
        r_vecs = positions_3d - positions_3d[i]
        dist_sq_sim = xp.einsum('ij,ij->i', r_vecs, r_vecs)

        # 避免与自身计算
        dist_sq_sim[i] = xp.inf
        
        # 计算牛顿引力
        dist_sq_m = dist_sq_sim * scale_sq
        # 使用 np.errstate 避免除零警告
        with xp.errstate(divide="ignore", invalid="ignore"):
            # 软化因子
            denominator = dist_sq_m + C.SOFTENING_FACTOR_SQ
            factor = g_constant / denominator
        
        # 将无用的值（inf, nan）替换为0
        factor[~xp.isfinite(factor)] = 0.0
        
        # 方向向量
        norm_r_vecs = r_vecs * (1.0 / xp.sqrt(dist_sq_sim))[:, xp.newaxis]
        norm_r_vecs[~xp.isfinite(norm_r_vecs)] = 0.0
        
        # 计算总牛顿加速度
        newtonian_acc = xp.sum(norm_r_vecs * (factor * masses)[:, xp.newaxis], axis=0)
        acc[i] += newtonian_acc

        # 计算广义相对论修正
        if use_gr_correction and velocities is not None:
            v_vec_i = velocities_3d[i]
            # 计算来自其他所有天体j的GR修正
            for j in range(n):
                if i == j: continue
                
                GM = g_constant * masses[j]
                r_vec_m = r_vecs[j] * C.SPACE_SCALE
                r_m_sq = dist_sq_m[j]
                
                if r_m_sq == 0: continue

                r_m = xp.sqrt(r_m_sq)
                c_sq = C.C_LIGHT ** 2
                
                # 计算角动量 L = r x v
                L_vec = xp.cross(r_vec_m, v_vec_i)
                L_sq = xp.dot(L_vec, L_vec)
                
                # 进动项 F_gr = - (3 * G * M * L^2) / (c^2 * r^4) * r_hat
                gr_acc_mag = (3 * GM * L_sq) / (r_m**4 * c_sq)

                gr_acc_vec = -gr_acc_mag * (r_vecs[j] / xp.sqrt(dist_sq_sim[j]))
                acc[i] += gr_acc_vec

    acc = acc[:, :positions.shape[1]]  # 返回与输入维度一致的加速度
    if use_gpu:
        acc = xp.asnumpy(acc)
    return acc

def rk4_step_arrays(
    positions,
    velocities,
    masses,
    fixed_mask,
    dt,
    g_constant,
    use_gr=False,
    *,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """使用RK4积分器。"""
    
    def deriv(pos, vel):
        return compute_accelerations(
            pos, masses, fixed_mask, g_constant, use_gr, vel, use_gpu=use_gpu
        )

    k1_v = deriv(positions, velocities)
    k1_p = velocities / C.SPACE_SCALE
    
    k2_v = deriv(positions + 0.5 * dt * k1_p, velocities + 0.5 * dt * k1_v)
    k2_p = (velocities + 0.5 * dt * k1_v) / C.SPACE_SCALE

    k3_v = deriv(positions + 0.5 * dt * k2_p, velocities + 0.5 * dt * k2_v)
    k3_p = (velocities + 0.5 * dt * k2_v) / C.SPACE_SCALE

    k4_v = deriv(positions + dt * k3_p, velocities + dt * k3_v)
    k4_p = (velocities + dt * k3_v) / C.SPACE_SCALE
    
    pos_new = positions + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    vel_new = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    pos_new[fixed_mask] = positions[fixed_mask]
    vel_new[fixed_mask] = velocities[fixed_mask]

    return pos_new, vel_new

def leapfrog_step_arrays(
    positions,
    velocities,
    masses,
    fixed_mask,
    dt,
    g_constant,
    use_gr=False,
    *,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """使用Leapfrog (kick-drift-kick)辛积分器推进模拟。"""
    
    # half kick
    accel_initial = compute_accelerations(
        positions, masses, fixed_mask, g_constant, use_gr, velocities, use_gpu=use_gpu
    )
    vel_half = velocities + accel_initial * (dt / 2.0)

    # full drift
    pos_new = positions + vel_half * dt / C.SPACE_SCALE

    # half kick
    accel_final = compute_accelerations(
        pos_new, masses, fixed_mask, g_constant, use_gr, vel_half, use_gpu=use_gpu
    )
    vel_new = vel_half + accel_final * (dt / 2.0)

    # 将固定天体的位置和速度重置
    pos_new[fixed_mask] = positions[fixed_mask]
    vel_new[fixed_mask] = velocities[fixed_mask]
    
    return pos_new, vel_new
