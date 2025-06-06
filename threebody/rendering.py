"""Rendering helpers and Body class."""
from collections import deque
import pygame
import pygame.gfxdraw
import numpy as np
from . import constants as C
from .jit import apply_boundary_conditions_jit

class Body:
    """Represents a celestial body with physical and visual properties."""
    ID_counter = 0

    def __init__(self, mass, x, y, vx, vy, color, radius, max_trail_length=C.DEFAULT_TRAIL_LENGTH,
                 fixed=False, name=None, show_trail=True):
        self.mass = float(mass)
        self.pos = np.array([float(x), float(y)], dtype=np.float64)
        self.vel = np.array([float(vx), float(vy)], dtype=np.float64)
        self.acc = np.zeros(2, dtype=np.float64)
        self.fixed = fixed
        self.color = color
        self.radius_pixels = max(1, int(radius))
        self.show_trail = show_trail
        self.max_trail_length = int(max_trail_length)
        self.trail = deque(maxlen=self.max_trail_length)
        self.visible = True
        self.id = Body.ID_counter
        Body.ID_counter += 1
        self.name = name if name else f"Body {self.id}"
        self.last_screen_pos = np.zeros(2)

    def update_physics_state(self, new_pos_sim, new_vel_m_s):
        if not self.fixed:
            self.pos = new_pos_sim
            self.vel = new_vel_m_s

    def update_trail(self, zoom, pan_offset):
        if not self.show_trail or not self.visible:
            if len(self.trail) > 0:
                self.trail.clear()
            return
        screen_pos = self.pos * zoom + pan_offset
        self.last_screen_pos = screen_pos
        self.trail.append(screen_pos.copy())

    def clear_trail(self):
        self.trail.clear()

    def set_trail_length(self, length):
        """Update maximum trail length and keep existing points."""
        self.max_trail_length = max(1, int(length))
        self.trail = deque(self.trail, maxlen=self.max_trail_length)

    def draw(self, screen, zoom, pan_offset, draw_labels):
        if not self.visible:
            return
        screen_pos = self.last_screen_pos
        draw_pos = (int(screen_pos[0]), int(screen_pos[1]))
        margin = 100
        sim_width_pixels = C.WIDTH - C.UI_SIDEBAR_WIDTH
        sim_height_pixels = C.HEIGHT - C.UI_BOTTOM_HEIGHT
        if (draw_pos[0] < -margin or draw_pos[0] > sim_width_pixels + margin or
                draw_pos[1] < -margin or draw_pos[1] > sim_height_pixels + margin):
            return
        if self.show_trail and len(self.trail) > 1:
            trail_points_pixels = list(self.trail)
            num_points = len(trail_points_pixels)
            max_segments = 100
            step = max(1, num_points // max_segments)
            for i in range(0, num_points - step, step):
                start_idx = i
                end_idx = i + step
                if end_idx >= num_points:
                    end_idx = num_points - 1
                if start_idx >= end_idx:
                    continue
                start_pos = (int(trail_points_pixels[start_idx][0]), int(trail_points_pixels[start_idx][1]))
                end_pos = (int(trail_points_pixels[end_idx][0]), int(trail_points_pixels[end_idx][1]))
                alpha = int(150 * (1.0 - (i / num_points)))
                alpha = max(0, min(255, alpha))
                if alpha > 10:
                    try:
                        pygame.draw.aaline(screen, (*self.color, alpha), start_pos, end_pos)
                    except TypeError:
                        pygame.draw.line(screen, self.color, start_pos, end_pos, 1)
                    except Exception:
                        pass
        effective_zoom_scale = max(0.1, zoom ** C.BODY_ZOOM_SCALING_POWER)
        draw_radius = max(3, int(self.radius_pixels * effective_zoom_scale))
        if (draw_pos[0] + draw_radius < 0 or draw_pos[0] - draw_radius > sim_width_pixels or
                draw_pos[1] + draw_radius < 0 or draw_pos[1] - draw_radius > sim_height_pixels):
            return
        if draw_radius > 4:
            glow_radius = draw_radius + int(draw_radius * 0.4)
            glow_alpha = 80
            glow_color = (*self.color, glow_alpha)
            try:
                max_r = glow_radius
                glow_surface = pygame.Surface((max_r * 2, max_r * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(glow_surface, max_r, max_r, max_r, glow_color)
                screen.blit(glow_surface, (draw_pos[0] - max_r, draw_pos[1] - max_r))
            except Exception:
                pass
        try:
            pygame.gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], draw_radius, self.color)
            pygame.gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], draw_radius, self.color)
        except Exception:
            pygame.draw.circle(screen, self.color, draw_pos, draw_radius)
        if draw_labels and draw_radius > 5:
            try:
                font_size = max(12, min(18, int(10 * zoom ** 0.5)))
                font = pygame.font.Font(None, font_size)
                label = font.render(self.name, True, C.WHITE)
                label_pos = (draw_pos[0] - label.get_width() // 2, draw_pos[1] + draw_radius + 2)
                screen.blit(label, label_pos)
            except Exception:
                pass

    def get_screen_pos(self, zoom, pan_offset):
        screen_pos = self.pos * zoom + pan_offset
        return (int(screen_pos[0]), int(screen_pos[1]))

    def handle_boundary_collision(self, bounds_sim, elasticity=0.8):
        if self.fixed:
            return
        new_pos, new_vel = apply_boundary_conditions_jit(self.pos, self.vel, bounds_sim, elasticity)
        if not np.array_equal(new_pos, self.pos) or not np.array_equal(new_vel, self.vel):
            self.pos = new_pos
            self.vel = new_vel


def render_gravitational_field(screen, bodies, g_constant, zoom, pan_offset):
    from .jit import NUMBA_AVAILABLE  # unused but keeps parity with original
    import matplotlib.cm as cm
    import numpy as np
    if not C.FIELD_RESOLUTION or not bodies:
        return
    resolution = C.FIELD_RESOLUTION
    sim_width_pixels = C.WIDTH - C.UI_SIDEBAR_WIDTH
    sim_height_pixels = C.HEIGHT - C.UI_BOTTOM_HEIGHT
    field_potential = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            screen_pos_x = (i + 0.5) * (sim_width_pixels / resolution)
            screen_pos_y = (j + 0.5) * (sim_height_pixels / resolution)
            screen_pos = np.array([screen_pos_x, screen_pos_y])
            world_pos_sim = (screen_pos - pan_offset) / (zoom + 1e-18)
            potential_at_point = 0.0
            for body in bodies:
                dist_vec_sim = body.pos - world_pos_sim
                dist_sq_sim = np.dot(dist_vec_sim, dist_vec_sim)
                if dist_sq_sim > 1e-18:
                    dist_meters = np.sqrt(dist_sq_sim) * C.SPACE_SCALE
                    if dist_meters > 0:
                        potential_at_point -= g_constant * body.mass / dist_meters
            field_potential[j, i] = potential_at_point
    min_pot = np.min(field_potential)
    max_pot = np.max(field_potential)
    if max_pot > min_pot:
        p_low, p_high = np.percentile(field_potential, [1, 99])
        if p_high <= p_low:
            p_high = p_low + 1e-9
        clipped_field = np.clip(field_potential, p_low, p_high)
        range_pot = p_high - p_low
        if range_pot == 0:
            range_pot = 1.0
        field_normalized = (clipped_field - p_low) / range_pot
    else:
        field_normalized = np.ones_like(field_potential) * 0.5
    cmap = cm.get_cmap('viridis')
    rgba_values = cmap(field_normalized.flatten())
    pixels = np.array(rgba_values[:, :3] * 255, dtype=np.uint8)
    heatmap_surface = pygame.Surface((resolution, resolution))
    pixel_index = 0
    for y in range(resolution):
        for x in range(resolution):
            heatmap_surface.set_at((x, y), tuple(pixels[pixel_index]))
            pixel_index += 1
    scaled_heatmap = pygame.transform.smoothscale(heatmap_surface, (sim_width_pixels, sim_height_pixels))
    scaled_heatmap.set_alpha(100)
    screen.blit(scaled_heatmap, (0, 0))

