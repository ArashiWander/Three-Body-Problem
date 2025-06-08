"""Physics helper functions for the simulation."""
import numpy as np
from . import constants as C
from .rendering import Body
from .jit import apply_boundary_conditions_jit


def calculate_system_energies(bodies, g_constant):
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
                dist_sim = np.sqrt(dist_sq_sim)
                dist_meters = dist_sim * C.SPACE_SCALE
                potential -= g_constant * body1.mass * body2.mass / dist_meters
    return kinetic, potential, kinetic + potential


def calculate_system_momentum(bodies):
    total_momentum = np.zeros(2, dtype=np.float64)
    for body in bodies:
        if body.fixed:
            continue
        total_momentum += body.mass * body.vel
    return total_momentum


def calculate_center_of_mass(bodies):
    total_mass = 0.0
    weighted_pos_sum = np.zeros(2, dtype=np.float64)
    weighted_vel_sum = np.zeros(2, dtype=np.float64)
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
            if len(fixed_bodies) > 0:
                com_pos_sim = sum(b.pos for b in fixed_bodies) / len(fixed_bodies)
            else:
                com_pos_sim = np.array([0.0, 0.0])
            com_vel_m_s = np.zeros(2)
            return com_pos_sim, com_vel_m_s
        return None, None
    com_pos_sim = weighted_pos_sum / total_mass
    com_vel_m_s = weighted_vel_sum / total_mass
    return com_pos_sim, com_vel_m_s


def calculate_accelerations_for_all(bodies, g_constant):
    accelerations = np.zeros((len(bodies), 2), dtype=np.float64)
    if len(bodies) < 2:
        return accelerations
    for i, body in enumerate(bodies):
        if body.fixed:
            continue
        acc_i = np.zeros(2, dtype=np.float64)
        for j, other_body in enumerate(bodies):
            if i == j:
                continue
            dist_vec_sim = other_body.pos - body.pos
            dist_sq_sim = np.dot(dist_vec_sim, dist_vec_sim)
            if dist_sq_sim == 0:
                continue
            dist_sq_meters = dist_sq_sim * (C.SPACE_SCALE ** 2)
            acc_mag = g_constant * other_body.mass / (dist_sq_meters + C.SOFTENING_FACTOR_SQ + 1e-18)
            dist_sim = np.sqrt(dist_sq_sim)
            direction = dist_vec_sim / dist_sim
            acc_i += direction * acc_mag
        accelerations[i] = acc_i
    return accelerations


def perform_rk4_step(bodies, dt, g_constant):
    n = len(bodies)
    if n == 0:
        return np.array([]), np.array([])
    initial_pos_sim = np.array([b.pos for b in bodies])
    initial_vel_m_s = np.array([b.vel for b in bodies])
    fixed_mask = np.array([b.fixed for b in bodies])
    k1_acc = calculate_accelerations_for_all(bodies, g_constant)
    k1_vel = initial_vel_m_s.copy()
    mid_pos_k2 = initial_pos_sim + (k1_vel * (0.5 * dt)) / C.SPACE_SCALE
    mid_vel_k2 = initial_vel_m_s + k1_acc * (0.5 * dt)
    temp_bodies_k2 = [{'pos': p, 'vel': v, 'mass': b.mass, 'fixed': b.fixed}
                      for p, v, b in zip(mid_pos_k2, mid_vel_k2, bodies)]
    k2_acc = calculate_accelerations_from_temp(temp_bodies_k2, g_constant)
    k2_vel = mid_vel_k2
    mid_pos_k3 = initial_pos_sim + (k2_vel * (0.5 * dt)) / C.SPACE_SCALE
    mid_vel_k3 = initial_vel_m_s + k2_acc * (0.5 * dt)
    temp_bodies_k3 = [{'pos': p, 'vel': v, 'mass': b.mass, 'fixed': b.fixed}
                      for p, v, b in zip(mid_pos_k3, mid_vel_k3, bodies)]
    k3_acc = calculate_accelerations_from_temp(temp_bodies_k3, g_constant)
    k3_vel = mid_vel_k3
    end_pos_k4 = initial_pos_sim + (k3_vel * dt) / C.SPACE_SCALE
    end_vel_k4 = initial_vel_m_s + k3_acc * dt
    temp_bodies_k4 = [{'pos': p, 'vel': v, 'mass': b.mass, 'fixed': b.fixed}
                      for p, v, b in zip(end_pos_k4, end_vel_k4, bodies)]
    k4_acc = calculate_accelerations_from_temp(temp_bodies_k4, g_constant)
    k4_vel = end_vel_k4
    final_pos_sim = initial_pos_sim.copy()
    final_vel_m_s = initial_vel_m_s.copy()
    pos_update = (dt / 6.0) * (k1_vel/C.SPACE_SCALE + 2*k2_vel/C.SPACE_SCALE + 2*k3_vel/C.SPACE_SCALE + k4_vel/C.SPACE_SCALE)
    vel_update = (dt / 6.0) * (k1_acc + 2*k2_acc + 2*k3_acc + k4_acc)
    final_pos_sim[~fixed_mask] += pos_update[~fixed_mask]
    final_vel_m_s[~fixed_mask] += vel_update[~fixed_mask]
    return final_pos_sim, final_vel_m_s


def calculate_accelerations_from_temp(temp_bodies_list, g_constant):
    num_bodies = len(temp_bodies_list)
    accelerations = np.zeros((num_bodies, 2), dtype=np.float64)
    if num_bodies < 2:
        return accelerations
    for i, current_tb in enumerate(temp_bodies_list):
        if current_tb['fixed']:
            continue
        acc_i = np.zeros(2, dtype=np.float64)
        for j, other_tb in enumerate(temp_bodies_list):
            if i == j:
                continue
            dist_vec_sim = other_tb['pos'] - current_tb['pos']
            dist_sq_sim = np.dot(dist_vec_sim, dist_vec_sim)
            if dist_sq_sim == 0:
                continue
            dist_sq_meters = dist_sq_sim * (C.SPACE_SCALE ** 2)
            acc_mag = g_constant * other_tb['mass'] / (dist_sq_meters + C.SOFTENING_FACTOR_SQ + 1e-18)
            dist_sim = np.sqrt(dist_sq_sim)
            direction = dist_vec_sim / dist_sim
            acc_i += direction * acc_mag
        accelerations[i] = acc_i
    return accelerations


def adaptive_rk4_step(bodies, current_dt, g_constant, error_tolerance, use_boundaries, bounds_sim):
    if not bodies:
        return 0.0, current_dt
    dt = max(C.MIN_TIME_STEP, min(current_dt, C.MAX_TIME_STEP))
    pos1, vel1 = perform_rk4_step(bodies, dt, g_constant)
    half_dt = dt / 2.0
    pos_half, vel_half = perform_rk4_step(bodies, half_dt, g_constant)
    temp_body_objects = []
    for i, b in enumerate(bodies):
        tb = Body(b.mass, 0, 0, 0, 0, b.color, b.radius_pixels, fixed=b.fixed)
        tb.pos = pos_half[i]
        tb.vel = vel_half[i]
        temp_body_objects.append(tb)
    pos2, vel2 = perform_rk4_step(temp_body_objects, half_dt, g_constant)
    max_rel_error = 0.0
    initial_pos_sim = np.array([b.pos for b in bodies])
    initial_vel_m_s = np.array([b.vel for b in bodies])
    for i, body in enumerate(bodies):
        if body.fixed:
            continue
        pos_error_sim = np.linalg.norm(pos2[i] - pos1[i])
        vel_error_m_s = np.linalg.norm(vel2[i] - vel1[i])
        pos_scale = np.linalg.norm(pos2[i]) + np.linalg.norm(initial_pos_sim[i]) + 1e-9 * C.SPACE_SCALE
        vel_scale = np.linalg.norm(vel2[i]) + np.linalg.norm(initial_vel_m_s[i]) + 1e-6
        rel_pos_error = pos_error_sim / pos_scale if pos_scale > 1e-15 else 0
        rel_vel_error = vel_error_m_s / vel_scale if vel_scale > 1e-15 else 0
        current_body_error = max(rel_pos_error, rel_vel_error)
        max_rel_error = max(max_rel_error, current_body_error)
    safety_factor = 0.9
    if max_rel_error <= 1e-15:
        scale_factor = 2.0
    else:
        scale_factor = safety_factor * (error_tolerance / max_rel_error) ** 0.2
    dt_new = dt * scale_factor
    dt_new = max(C.MIN_TIME_STEP, min(dt_new, C.MAX_TIME_STEP))
    if max_rel_error <= error_tolerance or dt <= C.MIN_TIME_STEP:
        for i, body in enumerate(bodies):
            if not body.fixed:
                body.update_physics_state(pos2[i], vel2[i])
                if use_boundaries and bounds_sim is not None:
                    body.handle_boundary_collision(bounds_sim)
        return dt, dt_new
    else:
        return 0.0, dt_new


def detect_and_handle_collisions(bodies, merge_on_collision=False):
    num_bodies = len(bodies)
    if num_bodies < 2:
        return []
    collided_pairs = set()
    indices_to_remove = []
    earth_radius_sim = C.EARTH_RADIUS_METERS / C.SPACE_SCALE
    physical_radii_sim = []
    for body in bodies:
        if body.mass <= 0:
            radius_sim = 0.001 * earth_radius_sim
        else:
            mass_ratio = body.mass / C.EARTH_MASS
            radius_sim = earth_radius_sim * (mass_ratio ** (1/3))
        physical_radii_sim.append(max(radius_sim, 0.001 * earth_radius_sim))
    for i in range(num_bodies):
        if i in indices_to_remove:
            continue
        for j in range(i + 1, num_bodies):
            if j in indices_to_remove:
                continue
            body1, body2 = bodies[i], bodies[j]
            radius1_sim, radius2_sim = physical_radii_sim[i], physical_radii_sim[j]
            distance_vec_sim = body2.pos - body1.pos
            dist_sq_sim = np.dot(distance_vec_sim, distance_vec_sim)
            collision_threshold = (radius1_sim + radius2_sim) * C.COLLISION_DISTANCE_FACTOR
            collision_threshold_sq = collision_threshold ** 2
            if dist_sq_sim < collision_threshold_sq and dist_sq_sim > 1e-18:
                pair = tuple(sorted((i, j)))
                if pair in collided_pairs:
                    continue
                dist_sim = np.sqrt(dist_sq_sim)
                if merge_on_collision:
                    if body1.fixed or body2.fixed:
                        continue
                    if body1.mass >= body2.mass:
                        survivor, removed = body1, body2
                        survivor_idx, removed_idx = i, j
                    else:
                        survivor, removed = body2, body1
                        survivor_idx, removed_idx = j, i
                    total_mass = survivor.mass + removed.mass
                    if total_mass == 0:
                        continue
                    new_vel = (survivor.mass * survivor.vel + removed.mass * removed.vel) / total_mass
                    new_pos = (survivor.mass * survivor.pos + removed.mass * removed.pos) / total_mass
                    survivor.mass = total_mass
                    survivor.pos = new_pos
                    survivor.vel = new_vel
                    survivor.radius_pixels = (body1.radius_pixels**3 + body2.radius_pixels**3)**(1/3)
                    survivor.name += f"+{removed.name}"
                    survivor.clear_trail()
                    if removed_idx not in indices_to_remove:
                        indices_to_remove.append(removed_idx)
                    collided_pairs.add(pair)
                else:
                    if body1.fixed and body2.fixed:
                        continue
                    normal_sim = distance_vec_sim / dist_sim
                    rel_vel_m_s = body2.vel - body1.vel
                    vel_along_normal = np.dot(rel_vel_m_s, normal_sim)
                    if vel_along_normal > 0:
                        continue
                    cor = 0.7
                    impulse = 0.0
                    if body1.fixed:
                        if body2.mass > 0:
                            impulse = -(1 + cor) * vel_along_normal / (1 / body2.mass)
                            body2.vel += (impulse / body2.mass) * normal_sim
                    elif body2.fixed:
                        if body1.mass > 0:
                            impulse = -(1 + cor) * vel_along_normal / (1 / body1.mass)
                            body1.vel -= (impulse / body1.mass) * normal_sim
                    else:
                        if body1.mass > 0 and body2.mass > 0:
                            inv_mass_sum = (1 / body1.mass) + (1 / body2.mass)
                            impulse = -(1 + cor) * vel_along_normal / inv_mass_sum
                            body1.vel -= (impulse / body1.mass) * normal_sim
                            body2.vel += (impulse / body2.mass) * normal_sim
                        elif body1.mass > 0:
                            impulse = -(1 + cor) * vel_along_normal / (1 / body1.mass)
                            body1.vel -= (impulse / body1.mass) * normal_sim
                        elif body2.mass > 0:
                            impulse = -(1 + cor) * vel_along_normal / (1 / body2.mass)
                            body2.vel += (impulse / body2.mass) * normal_sim
                    overlap_sim = collision_threshold - dist_sim
                    if overlap_sim > 0:
                        correction_factor = 0.6
                        correction_vec = normal_sim * overlap_sim * correction_factor
                        if body1.fixed:
                            body2.pos += correction_vec * 2
                        elif body2.fixed:
                            body1.pos -= correction_vec * 2
                        else:
                            m1 = body1.mass
                            m2 = body2.mass
                            total_mass = m1 + m2
                            if total_mass > 0:
                                body1.pos -= correction_vec * (m2 / total_mass) * 2
                                body2.pos += correction_vec * (m1 / total_mass) * 2
                            elif m1 > 0:
                                body2.pos += correction_vec * 2
                            elif m2 > 0:
                                body1.pos -= correction_vec * 2
                    collided_pairs.add(pair)
    return sorted(indices_to_remove, reverse=True)


def get_world_bounds_sim(zoom, pan_offset):
    sim_width_pixels = C.WIDTH - C.UI_SIDEBAR_WIDTH
    sim_height_pixels = C.HEIGHT - C.UI_BOTTOM_HEIGHT
    min_screen = np.array([0, 0])
    min_world_sim = (min_screen - pan_offset) / (zoom + 1e-18)
    max_screen = np.array([sim_width_pixels, sim_height_pixels])
    max_world_sim = (max_screen - pan_offset) / (zoom + 1e-18)
    return (min_world_sim[0], min_world_sim[1], max_world_sim[0], max_world_sim[1])
