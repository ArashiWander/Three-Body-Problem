import numpy as np
from . import constants as C
from .integrators import rk4_step_arrays, leapfrog_step_arrays, symplectic4_step_arrays
from .physics import Body as PhysicsBody
from .jit import apply_boundary_conditions_jit


class _QuadTree:
    """Simple 2-D quadtree for fast neighbourhood queries."""

    __slots__ = (
        "bounds",
        "depth",
        "capacity",
        "max_depth",
        "points",
        "children",
    )

    def __init__(self, bounds, depth=0, capacity=8, max_depth=10):
        self.bounds = bounds  # (xmin, ymin, xmax, ymax)
        self.depth = depth
        self.capacity = capacity
        self.max_depth = max_depth
        self.points = []  # list of (index, (x,y))
        self.children = None

    def _subdivide(self):
        x0, y0, x1, y1 = self.bounds
        hx = (x0 + x1) / 2
        hy = (y0 + y1) / 2
        self.children = [
            _QuadTree((x0, y0, hx, hy), self.depth + 1, self.capacity, self.max_depth),
            _QuadTree((hx, y0, x1, hy), self.depth + 1, self.capacity, self.max_depth),
            _QuadTree((x0, hy, hx, y1), self.depth + 1, self.capacity, self.max_depth),
            _QuadTree((hx, hy, x1, y1), self.depth + 1, self.capacity, self.max_depth),
        ]

    def _insert_into_children(self, idx, point):
        for child in self.children:
            if child._contains(point):
                child.insert(idx, point)
                return
        # Fallback if point lies on boundary due to precision errors
        self.children[0].insert(idx, point)

    def _contains(self, point):
        x0, y0, x1, y1 = self.bounds
        x, y = point
        return x0 <= x <= x1 and y0 <= y <= y1

    def insert(self, idx, point):
        if self.children is not None:
            self._insert_into_children(idx, point)
            return
        self.points.append((idx, point))
        if len(self.points) > self.capacity and self.depth < self.max_depth:
            self._subdivide()
            for i, p in self.points:
                self._insert_into_children(i, p)
            self.points.clear()

    def _intersects(self, region):
        x0, y0, x1, y1 = self.bounds
        rx0, ry0, rx1, ry1 = region
        return not (rx1 < x0 or rx0 > x1 or ry1 < y0 or ry0 > y1)

    def query(self, region, out=None):
        if out is None:
            out = []
        if not self._intersects(region):
            return out
        x0, y0, x1, y1 = region
        for idx, (x, y) in self.points:
            if x0 <= x <= x1 and y0 <= y <= y1:
                out.append(idx)
        if self.children is not None:
            for child in self.children:
                child.query(region, out)
        return out

def step_simulation(
    bodies,
    dt,
    g_constant,
    integrator_type='Symplectic',
    use_gr=False,
    *,
    use_gpu: bool = False,
):
    """
    根据选择的积分器类型推进模拟。

    这是物理更新的核心调用函数，它会根据UI中的选择，
    分派任务给不同的积分器（RK4或Leapfrog）。
    """
    if not bodies:
        return

    positions = np.array([b.pos for b in bodies])
    velocities = np.array([b.vel for b in bodies])
    masses = np.array([b.mass for b in bodies])
    fixed_mask = np.array([b.fixed for b in bodies])

    if integrator_type == 'RK4':
        new_pos, new_vel = rk4_step_arrays(
            positions,
            velocities,
            masses,
            fixed_mask,
            dt,
            g_constant,
            use_gr,
            use_gpu=use_gpu,
        )
    elif integrator_type == 'Symplectic4':
        new_pos, new_vel = symplectic4_step_arrays(
            positions,
            velocities,
            masses,
            fixed_mask,
            dt,
            g_constant,
            use_gr,
            use_gpu=use_gpu,
        )
    else:  # 默认为 Symplectic (Leapfrog)
        new_pos, new_vel = leapfrog_step_arrays(
            positions,
            velocities,
            masses,
            fixed_mask,
            dt,
            g_constant,
            use_gr,
            use_gpu=use_gpu,
        )

    for i, body in enumerate(bodies):
        if body.fixed:
            continue
        if hasattr(body, "update_physics_state"):
            body.update_physics_state(new_pos[i], new_vel[i])
        else:
            body.pos = new_pos[i]
            body.vel = new_vel[i]

# --- 以下是所有函数的完整实现 ---

def calculate_system_energies(bodies, g_constant):
    """计算系统的动能、势能和总能量。"""
    kinetic = 0.0
    potential = 0.0
    for body in bodies:
        if body.fixed:
            continue
        speed_sq = np.dot(body.vel, body.vel)
        kinetic += 0.5 * body.mass * speed_sq
        
    num_bodies = len(bodies)
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            body1, body2 = bodies[i], bodies[j]
            distance_vec_sim = body2.pos - body1.pos
            dist_sq_sim = np.dot(distance_vec_sim, distance_vec_sim)
            if dist_sq_sim > 1e-18:
                dist_meters = np.sqrt(dist_sq_sim) * C.SPACE_SCALE
                potential -= g_constant * body1.mass * body2.mass / dist_meters
    return kinetic, potential, kinetic + potential


def calculate_center_of_mass(bodies):
    """计算系统的质心位置和速度。"""
    total_mass = 0.0
    weighted_pos_sum = np.zeros(3, dtype=np.float64)
    weighted_vel_sum = np.zeros(3, dtype=np.float64)
    has_mass = False
    for body in bodies:
        if not body.fixed and body.mass > 0:
            has_mass = True
            total_mass += body.mass
            weighted_pos_sum += body.pos * body.mass
            weighted_vel_sum += body.vel * body.mass
    
    if not has_mass or total_mass == 0:
        fixed_bodies = [b for b in bodies if b.fixed]
        if fixed_bodies:
            com_pos_sim = sum(b.pos for b in fixed_bodies) / len(fixed_bodies)
            com_vel_m_s = np.zeros(3)
            return com_pos_sim, com_vel_m_s
        return None, None
        
    com_pos_sim = weighted_pos_sum / total_mass
    com_vel_m_s = weighted_vel_sum / total_mass
    return com_pos_sim, com_vel_m_s

def perform_rk4_step(bodies, dt, g_constant, *, use_gpu: bool = False):
    """对一组天体执行一个RK4步长。（为自适应步长提供的辅助函数）"""
    if not bodies:
        return np.array([]), np.array([])

    positions = np.array([b.pos for b in bodies])
    velocities = np.array([b.vel for b in bodies])
    masses = np.array([b.mass for b in bodies])
    fixed_mask = np.array([b.fixed for b in bodies])

    # 注意：这里的调用未使用GR修正，因为自适应步长主要用于牛顿力学
    return rk4_step_arrays(
        positions,
        velocities,
        masses,
        fixed_mask,
        dt,
        g_constant,
        use_gr=False,
        use_gpu=use_gpu,
    )


def adaptive_rk4_step(
    bodies,
    current_dt,
    g_constant,
    error_tolerance,
    use_boundaries,
    bounds_sim,
    *,
    use_gpu: bool = False,
):
    """执行一个自适应RK4步长，并返回建议的下一个步长。"""
    if not bodies:
        return 0.0, current_dt
        
    dt = max(C.MIN_TIME_STEP, min(current_dt, C.MAX_TIME_STEP))
    
    # 执行一个完整步长
    pos1, vel1 = perform_rk4_step(bodies, dt, g_constant, use_gpu=use_gpu)
    
    # 执行两个半步长
    pos_half, vel_half = perform_rk4_step(bodies, dt / 2.0, g_constant, use_gpu=use_gpu)
    temp_bodies = [PhysicsBody(b.mass, pos_half[i], vel_half[i], fixed=b.fixed) for i, b in enumerate(bodies)]
    pos2, vel2 = perform_rk4_step(temp_bodies, dt / 2.0, g_constant, use_gpu=use_gpu)
    
    # 计算误差
    max_rel_error = 0.0
    initial_pos_sim = np.array([b.pos for b in bodies])
    
    for i, body in enumerate(bodies):
        if body.fixed:
            continue
        pos_error_sim = np.linalg.norm(pos2[i] - pos1[i])
        pos_scale = np.linalg.norm(pos2[i]) + np.linalg.norm(initial_pos_sim[i]) + 1e-9
        rel_pos_error = pos_error_sim / pos_scale if pos_scale > 1e-15 else 0
        max_rel_error = max(max_rel_error, rel_pos_error)

    # 根据误差调整时间步长
    safety_factor = 0.9
    if max_rel_error <= 1e-15:
        scale_factor = 2.0
    else:
        scale_factor = safety_factor * (error_tolerance / max_rel_error) ** 0.2
        
    dt_new = dt * scale_factor
    dt_new = max(C.MIN_TIME_STEP, min(dt_new, C.MAX_TIME_STEP))

    # 如果误差在容忍范围内，则接受步长
    if max_rel_error <= error_tolerance or dt <= C.MIN_TIME_STEP:
        for i, body in enumerate(bodies):
            if body.fixed:
                continue
            if hasattr(body, "update_physics_state"):
                body.update_physics_state(pos2[i], vel2[i])
            else:
                body.pos = pos2[i]
                body.vel = vel2[i]
            if use_boundaries and bounds_sim is not None and hasattr(body, "handle_boundary_collision"):
                body.handle_boundary_collision(bounds_sim)
        return dt, dt_new
    else:
        # 否则，拒绝步长并建议使用更小的时间步重试
        return 0.0, dt_new


def detect_and_handle_collisions(bodies, merge_on_collision=False):
    """检测并处理天体之间的碰撞。"""
    num_bodies = len(bodies)
    if num_bodies < 2:
        return []

    collided_pairs: set[tuple[int, int]] = set()
    indices_to_remove: list[int] = []

    earth_radius_sim = C.EARTH_RADIUS_METERS / C.SPACE_SCALE
    physical_radii_sim = []
    for body in bodies:
        if body.mass <= 0:
            radius_sim = 0.001 * earth_radius_sim
        else:
            mass_ratio = body.mass / C.EARTH_MASS
            radius_sim = earth_radius_sim * (mass_ratio ** (1 / 3))
        radius_sim *= C.COLLISION_DISTANCE_FACTOR
        physical_radii_sim.append(max(radius_sim, 0.001 * earth_radius_sim))

    # Build quadtree using 2D positions
    positions_2d = np.array([b.pos[:2] for b in bodies], dtype=float)
    xmin, ymin = positions_2d.min(axis=0)
    xmax, ymax = positions_2d.max(axis=0)
    pad = 1e-9
    root = _QuadTree((xmin - pad, ymin - pad, xmax + pad, ymax + pad))
    for idx, pt in enumerate(positions_2d):
        root.insert(idx, tuple(pt))

    max_radius = max(physical_radii_sim)

    for i in range(num_bodies):
        if i in indices_to_remove:
            continue
        body1 = bodies[i]
        radius1_sim = physical_radii_sim[i]
        query_radius = radius1_sim + max_radius
        region = (
            positions_2d[i, 0] - query_radius,
            positions_2d[i, 1] - query_radius,
            positions_2d[i, 0] + query_radius,
            positions_2d[i, 1] + query_radius,
        )
        candidates = root.query(region)
        for j in candidates:
            if j <= i or j in indices_to_remove:
                continue
            pair = tuple(sorted((i, j)))
            if pair in collided_pairs:
                continue

            body2 = bodies[j]
            radius2_sim = physical_radii_sim[j]
            distance_vec_sim = body2.pos - body1.pos
            dist_sq_sim = float(np.dot(distance_vec_sim, distance_vec_sim))
            collision_threshold_sq = (radius1_sim + radius2_sim) ** 2

            if dist_sq_sim < collision_threshold_sq and dist_sq_sim > 1e-18:
                dist_sim = float(np.sqrt(dist_sq_sim))

                if merge_on_collision:
                    if body1.fixed or body2.fixed:
                        continue

                    survivor, removed = (
                        (body1, body2) if body1.mass >= body2.mass else (body2, body1)
                    )
                    removed_idx = j if body1.mass >= body2.mass else i

                    total_mass = survivor.mass + removed.mass
                    if total_mass == 0:
                        continue

                    new_vel = (
                        survivor.mass * survivor.vel + removed.mass * removed.vel
                    ) / total_mass
                    new_pos = (
                        survivor.mass * survivor.pos + removed.mass * removed.pos
                    ) / total_mass

                    survivor.mass = total_mass
                    survivor.pos = new_pos
                    survivor.vel = new_vel

                    if hasattr(survivor, "radius_pixels"):
                        survivor.radius_pixels = (
                            body1.radius_pixels ** 3 + body2.radius_pixels ** 3
                        ) ** (1 / 3)
                    if hasattr(survivor, "name"):
                        survivor.name += f"+{removed.name}"
                    if hasattr(survivor, "clear_trail"):
                        survivor.clear_trail()

                    if removed_idx not in indices_to_remove:
                        indices_to_remove.append(removed_idx)
                    collided_pairs.add(pair)
                else:  # Bounce logic
                    if body1.fixed and body2.fixed:
                        continue

                    normal_sim = distance_vec_sim / dist_sim
                    rel_vel = body2.vel - body1.vel
                    vel_along_normal = float(np.dot(rel_vel, normal_sim))

                    if vel_along_normal > 0:
                        continue

                    cor = 0.7  # 恢复系数

                    if body1.fixed:
                        impulse = -(1 + cor) * vel_along_normal * body2.mass
                        body2.vel += (impulse / body2.mass) * normal_sim
                    elif body2.fixed:
                        impulse = -(1 + cor) * vel_along_normal * body1.mass
                        body1.vel -= (impulse / body1.mass) * normal_sim
                    else:
                        if body1.mass > 0 and body2.mass > 0:
                            inv_mass_sum = 1 / body1.mass + 1 / body2.mass
                            impulse = -(1 + cor) * vel_along_normal / inv_mass_sum
                            body1.vel -= (impulse / body1.mass) * normal_sim
                            body2.vel += (impulse / body2.mass) * normal_sim

                    # 位置修正，防止天体卡住
                    overlap_sim = np.sqrt(collision_threshold_sq) - dist_sim
                    if overlap_sim > 0:
                        correction_factor = 0.8
                        correction_vec = normal_sim * overlap_sim * correction_factor
                        if body1.fixed:
                            body2.pos += correction_vec * 2
                        elif body2.fixed:
                            body1.pos -= correction_vec * 2
                        else:
                            total_mass = body1.mass + body2.mass
                            if total_mass > 0:
                                body1.pos -= (
                                    correction_vec * (body2.mass / total_mass) * 2
                                )
                                body2.pos += (
                                    correction_vec * (body1.mass / total_mass) * 2
                                )

                    collided_pairs.add(pair)

    return sorted(indices_to_remove, reverse=True)


def get_world_bounds_sim(zoom, pan_offset):
    """根据相机状态获取世界边界（模拟单位）。"""
    sim_width_pixels = C.WIDTH - C.UI_SIDEBAR_WIDTH
    sim_height_pixels = C.HEIGHT - C.UI_BOTTOM_HEIGHT
    min_screen = np.array([0, 0])
    max_screen = np.array([sim_width_pixels, sim_height_pixels])
    
    min_world_sim = (min_screen - pan_offset) / (zoom + 1e-18)
    max_world_sim = (max_screen - pan_offset) / (zoom + 1e-18)
    
    return (min_world_sim[0], min_world_sim[1], max_world_sim[0], max_world_sim[1])
