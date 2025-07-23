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
        return xp.zeros_like(positions, dtype=xp.float64)

    # 确保位置和速度是3D的
    if positions.shape[1] == 2:
        positions_3d = xp.hstack([
            positions,
            xp.zeros((n, 1), dtype=positions.dtype),
        ])
    else:
        positions_3d = positions

    if velocities is not None and velocities.shape[1] == 2:
        velocities_3d = xp.hstack([
            velocities,
            xp.zeros((n, 1), dtype=velocities.dtype),
        ])
    else:
        velocities_3d = velocities

    acc = xp.zeros_like(positions_3d, dtype=xp.float64)
    scale_sq = C.SPACE_SCALE ** 2

    # 使用严格的成对力计算确保牛顿第三定律精确满足
    for i in range(n):
        if fixed_mask[i]:
            continue
        for j in range(i + 1, n):
            if fixed_mask[j]:
                continue

            # 计算天体i和j之间的相对位置向量 (从i指向j)
            r_vec_ij = positions_3d[j] - positions_3d[i]
            dist_sq_sim = xp.sum(r_vec_ij ** 2)

            # 跳过距离为零的情况
            if dist_sq_sim == 0:
                continue

            # 转换到米制距离
            dist_sq_m = dist_sq_sim * scale_sq

            # 软化因子
            softening_sq = xp.maximum(C.SOFTENING_FACTOR_SQ, 1e-20 * scale_sq)

            # 计算引力加速度的大小
            with xp.errstate(divide="ignore", invalid="ignore"):
                denominator = dist_sq_m + softening_sq
                if denominator == 0 or not xp.isfinite(denominator):
                    continue

                # a_i = G * m_j / r²  (j对i的加速度)
                # a_j = G * m_i / r²  (i对j的加速度)
                accel_mag_i = g_constant * masses[j] / denominator
                accel_mag_j = g_constant * masses[i] / denominator

            if not (xp.isfinite(accel_mag_i) and xp.isfinite(accel_mag_j)):
                continue

            # 单位方向向量
            dist_sim = xp.sqrt(dist_sq_sim)
            r_hat_ij = r_vec_ij / dist_sim  # 从i指向j的单位向量

            # 牛顿第三定律：天体间的力大小相等方向相反
            # 天体j对天体i的加速度（指向j，即引力）
            acc[i] += accel_mag_i * r_hat_ij
            # 天体i对天体j的加速度（指向i，即引力）
            acc[j] -= accel_mag_j * r_hat_ij  # 注意负号

    # 广义相对论修正（简化版，仅当启用时）
    if use_gr_correction and velocities is not None:
        for i in range(n):
            if fixed_mask[i]:
                continue

            v_vec_i = velocities_3d[i]
            for j in range(n):
                if i == j or fixed_mask[j]:
                    continue

                r_vec_m = (positions_3d[j] - positions_3d[i]) * C.SPACE_SCALE
                r_sq_m = xp.sum(r_vec_m ** 2)

                if r_sq_m == 0:
                    continue

                L_vec = xp.cross(r_vec_m, v_vec_i)
                L_sq = xp.sum(L_vec ** 2)

                with xp.errstate(divide="ignore", invalid="ignore"):
                    gr_factor = -3.0 * g_constant * masses[j] * L_sq / (
                        (C.C_LIGHT ** 2) * (r_sq_m ** 2)
                    )

                if xp.isfinite(gr_factor):
                    r_hat_m = r_vec_m / xp.sqrt(r_sq_m)
                    acc[i] += gr_factor * r_hat_m

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


def symplectic4_step_arrays(
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
    """Fourth-order symplectic integrator using Yoshida coefficients."""

    a1 = 0.5153528374311229
    a2 = -0.08578201941297365
    a3 = 0.4415830236164665
    a4 = 0.1288461583653842

    b1 = 0.1344961992774311
    b2 = -0.2248198030794208
    b3 = 0.7562300005156683
    b4 = 0.3340036032863214

    pos = positions
    vel = velocities

    acc = compute_accelerations(pos, masses, fixed_mask, g_constant, use_gr, vel, use_gpu=use_gpu)
    vel = vel + b1 * dt * acc
    pos = pos + a1 * dt * vel / C.SPACE_SCALE

    acc = compute_accelerations(pos, masses, fixed_mask, g_constant, use_gr, vel, use_gpu=use_gpu)
    vel = vel + b2 * dt * acc
    pos = pos + a2 * dt * vel / C.SPACE_SCALE

    acc = compute_accelerations(pos, masses, fixed_mask, g_constant, use_gr, vel, use_gpu=use_gpu)
    vel = vel + b3 * dt * acc
    pos = pos + a3 * dt * vel / C.SPACE_SCALE

    acc = compute_accelerations(pos, masses, fixed_mask, g_constant, use_gr, vel, use_gpu=use_gpu)
    vel = vel + b4 * dt * acc
    pos = pos + a4 * dt * vel / C.SPACE_SCALE

    pos[fixed_mask] = positions[fixed_mask]
    vel[fixed_mask] = velocities[fixed_mask]

    return pos, vel


def forest_ruth_step_arrays(
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
    """Fourth-order symplectic integrator using the Forest–Ruth scheme."""

    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))

    pos, vel = leapfrog_step_arrays(
        positions,
        velocities,
        masses,
        fixed_mask,
        w1 * dt,
        g_constant,
        use_gr,
        use_gpu=use_gpu,
    )

    pos, vel = leapfrog_step_arrays(
        pos,
        vel,
        masses,
        fixed_mask,
        w0 * dt,
        g_constant,
        use_gr,
        use_gpu=use_gpu,
    )

    pos, vel = leapfrog_step_arrays(
        pos,
        vel,
        masses,
        fixed_mask,
        w1 * dt,
        g_constant,
        use_gr,
        use_gpu=use_gpu,
    )

    return pos, vel
