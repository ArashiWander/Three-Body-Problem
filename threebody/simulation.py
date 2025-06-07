import numpy as np
import pygame
import pygame.gfxdraw

from importlib.metadata import version, PackageNotFoundError

from . import constants as C

try:
    _PACKAGE_VERSION = version("threebody")
except PackageNotFoundError:
    _PACKAGE_VERSION = "0.0.0"
from .presets import PRESETS
from .rendering import Body, render_gravitational_field
from .physics_utils import (
    calculate_center_of_mass,
    perform_rk4_step,
    adaptive_rk4_step,
    detect_and_handle_collisions,
    get_world_bounds_sim,
)


class Simulation:
    """Interactive n-body simulation wrapper."""

    def __init__(self, init_pygame: bool = True):
        self.bodies = []
        self.current_preset = "Sun & Earth"
        self.simulation_time = 0.0
        self.time_step = C.TIME_STEP_BASE
        self.paused = False
        self.running = False

        self.current_zoom = C.ZOOM_BASE
        self.current_pan = C.INITIAL_PAN_OFFSET.copy()
        self.target_zoom = C.ZOOM_BASE
        self.target_pan = C.INITIAL_PAN_OFFSET.copy()

        self.show_trails = C.SHOW_TRAILS
        self.show_grav_field = C.SHOW_GRAV_FIELD
        self.adaptive_stepping = C.ADAPTIVE_STEPPING
        self.speed_factor = C.SPEED_FACTOR

        if init_pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
            pygame.display.set_caption(f"N-Body Simulation v{__version__}")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

    # ------------------------------------------------------------------
    def load_preset(self, preset_name: str) -> None:
        """Load bodies defined in a named preset."""
        if preset_name not in PRESETS:
            raise KeyError(f"Preset '{preset_name}' not found")
        self.bodies = []
        Body.ID_counter = 0
        for cfg in PRESETS[preset_name]:
            self.bodies.append(
                Body(
                    cfg.get("mass", C.EARTH_MASS),
                    cfg.get("x", 0.0),
                    cfg.get("y", 0.0),
                    cfg.get("vx", 0.0),
                    cfg.get("vy", 0.0),
                    cfg.get("color", C.WHITE),
                    cfg.get("radius", 5),
                    fixed=cfg.get("fixed", False),
                    name=cfg.get("name"),
                    show_trail=self.show_trails,
                )
            )
        self.simulation_time = 0.0
        self.time_step = C.TIME_STEP_BASE
        self.current_preset = preset_name
        self.target_zoom = self.current_zoom = C.ZOOM_BASE
        com, _ = calculate_center_of_mass(self.bodies)
        if com is not None:
            center = np.array(
                [(C.WIDTH - C.UI_SIDEBAR_WIDTH) / 2, (C.HEIGHT - C.UI_BOTTOM_HEIGHT) / 2]
            )
            self.target_pan = self.current_pan = center - com * self.target_zoom
        else:
            self.target_pan = self.current_pan = C.INITIAL_PAN_OFFSET.copy()

    # ------------------------------------------------------------------
    def handle_events(self) -> None:
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.load_preset(self.current_preset)

    # ------------------------------------------------------------------
    def update_physics(self) -> None:
        """Advance physics and trails."""
        if self.paused or not self.bodies:
            return

        g_const = C.INITIAL_G
        sim_dt = self.time_step * self.speed_factor
        bounds_sim = get_world_bounds_sim(self.current_zoom, self.current_pan)

        if self.adaptive_stepping:
            advanced, suggestion = adaptive_rk4_step(
                self.bodies,
                sim_dt,
                g_const,
                C.ERROR_TOLERANCE,
                False,
                bounds_sim,
            )
            self.time_step = suggestion
            self.simulation_time += advanced
        else:
            new_pos, new_vel = perform_rk4_step(self.bodies, sim_dt, g_const)
            for i, body in enumerate(self.bodies):
                if not body.fixed:
                    body.update_physics_state(new_pos[i], new_vel[i])
            self.simulation_time += sim_dt

        detect_and_handle_collisions(self.bodies, merge_on_collision=C.MERGE_ON_COLLISION)
        for body in self.bodies:
            body.update_trail(self.current_zoom, self.current_pan)

    # ------------------------------------------------------------------
    def draw(self) -> None:
        """Render the current frame."""
        if self.screen is None:
            return
        self.screen.fill(C.BLACK)
        if self.show_grav_field:
            render_gravitational_field(
                self.screen, self.bodies, C.INITIAL_G, self.current_zoom, self.current_pan
            )
        for body in self.bodies[: C.MAX_DISPLAY_BODIES]:
            body.draw(self.screen, self.current_zoom, self.current_pan, draw_labels=False)
        pygame.display.flip()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main application loop."""
        if self.screen is None or self.clock is None:
            raise RuntimeError("Simulation cannot run without pygame initialized")
        self.load_preset(self.current_preset)
        self.running = True
        while self.running:
            self.clock.tick(60)
            self.handle_events()
            self.current_zoom += (self.target_zoom - self.current_zoom) * C.CAMERA_SMOOTHING
            self.current_pan += (self.target_pan - self.current_pan) * C.CAMERA_SMOOTHING
            self.update_physics()
            self.draw()
        pygame.quit()
