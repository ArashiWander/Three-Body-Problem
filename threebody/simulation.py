# Simulation class extracted from simulation_full.py
import pygame
import numpy as np
from collections import deque
from .constants import *
from .utils import mass_to_display, time_to_display
from .presets import PRESETS
from .rendering import Body, render_gravitational_field
from .physics_utils import (
    calculate_center_of_mass,
    perform_rk4_step,
    adaptive_rk4_step,
    detect_and_handle_collisions,
    get_world_bounds_sim,
)
import pygame_gui

class Simulation:
    """Encapsulates the main simulation loop and helpers."""

    def __init__(self):
        # Flags from constants
        self.show_trails = SHOW_TRAILS
        self.show_grav_field = SHOW_GRAV_FIELD
        self.adaptive_stepping = ADAPTIVE_STEPPING
        self.speed_factor = SPEED_FACTOR

        # Pygame/UI setup
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"N-Body Simulation v{VERSION}")
        self.clock = pygame.time.Clock()
        self.ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT))
        self._build_ui()
        self._init_state()

    def _build_ui(self):
        """Create UI panels and widgets."""
        self.control_panel_rect = pygame.Rect(
            (WIDTH - UI_SIDEBAR_WIDTH, 0), (UI_SIDEBAR_WIDTH, HEIGHT)
        )
        self.status_bar_rect = pygame.Rect(
            (0, HEIGHT - UI_BOTTOM_HEIGHT), (WIDTH - UI_SIDEBAR_WIDTH, UI_BOTTOM_HEIGHT)
        )
        self.control_panel = pygame_gui.elements.UIPanel(
            relative_rect=self.control_panel_rect, manager=self.ui_manager
        )
        self.status_bar = pygame_gui.elements.UIPanel(
            relative_rect=self.status_bar_rect, manager=self.ui_manager
        )
        y_pos = 10
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)),
            text="N-Body Simulator",
            manager=self.ui_manager,
            container=self.control_panel,
            object_id="#title_label",
        )
        y_pos += 40
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (80, 30)),
            text="Preset:",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        self.preset_dropdown = pygame_gui.elements.UIDropDownMenu(
            options_list=list(PRESETS.keys()),
            starting_option="Sun & Earth",
            relative_rect=pygame.Rect((100, y_pos), (UI_SIDEBAR_WIDTH - 120, 30)),
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 40
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Simulation",
            manager=self.ui_manager,
            container=self.control_panel,
            object_id="#section_header",
        )
        y_pos += 25
        self.play_pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)),
            text="Pause",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(((UI_SIDEBAR_WIDTH - 30) // 2 + 20, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)),
            text="Reset",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 40
        self.speed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text=f"Speed: {self.speed_factor:.1f}x",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 20
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            start_value=self.speed_factor,
            value_range=(0.05, 5.0),
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 30
        self.gravity_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Gravity: 10.0x",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 20
        self.gravity_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            start_value=10.0,
            value_range=(0.0, 100.0),
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 30
        self.adaptive_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)),
            text=f"Adaptive Step: {'ON' if self.adaptive_stepping else 'OFF'}",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 40
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Visualization",
            manager=self.ui_manager,
            container=self.control_panel,
            object_id="#section_header",
        )
        y_pos += 25
        self.trails_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)),
            text=f"Trails: {'ON' if self.show_trails else 'OFF'}",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        self.gfield_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(((UI_SIDEBAR_WIDTH - 30) // 2 + 20, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)),
            text="Grav Field: OFF",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 40
        self.boundaries_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)),
            text="Boundaries: ON",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 40
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Camera",
            manager=self.ui_manager,
            container=self.control_panel,
            object_id="#section_header",
        )
        y_pos += 25
        self.center_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)),
            text="Center View",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        self.zoom_reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(((UI_SIDEBAR_WIDTH - 30) // 2 + 20, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)),
            text="Reset Zoom",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 40
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Add Body",
            manager=self.ui_manager,
            container=self.control_panel,
            object_id="#section_header",
        )
        y_pos += 25
        self.mass_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text=f"Mass: {mass_to_display(DEFAULT_NEXT_BODY_MASS)}",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 20
        mass_min_log = math.log10(max(1e18, 0.001 * EARTH_MASS))
        mass_max_log = math.log10(100 * SOLAR_MASS)
        mass_default_log = math.log10(max(1e-9, DEFAULT_NEXT_BODY_MASS))
        slider_default = 0.5
        if mass_max_log > mass_min_log:
            slider_default = (mass_default_log - mass_min_log) / (mass_max_log - mass_min_log)
            slider_default = max(0.0, min(1.0, slider_default))
        self.add_mass_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            start_value=slider_default,
            value_range=(0.0, 1.0),
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 30
        self.radius_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text=f"Radius: {DEFAULT_NEXT_BODY_RADIUS_PIXELS} px",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 20
        self.add_radius_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            start_value=DEFAULT_NEXT_BODY_RADIUS_PIXELS,
            value_range=(1, 50),
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 30
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 40)),
            text="Right-click & drag in view\nto add body with velocity.",
            manager=self.ui_manager,
            container=self.control_panel,
            object_id="#instruction_label",
        )
        y_pos += 50
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Selected Body",
            manager=self.ui_manager,
            container=self.control_panel,
            object_id="#section_header",
        )
        y_pos += 25
        self.selected_body_name_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Name: None",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 25
        self.selected_body_mass_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            text="Mass: N/A",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        y_pos += 20
        self.selected_mass_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
            start_value=0.5,
            value_range=(0.0, 1.0),
            manager=self.ui_manager,
            container=self.control_panel,
        )
        self.selected_mass_slider.disable()
        y_pos += 40
        self.help_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)),
            text="Help / Controls",
            manager=self.ui_manager,
            container=self.control_panel,
        )
        self.status_text_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, 0), (WIDTH - UI_SIDEBAR_WIDTH - 20, UI_BOTTOM_HEIGHT)),
            text="Status: Initializing...",
            manager=self.ui_manager,
            container=self.status_bar,
        )
        help_rect = pygame.Rect(0, 0, (WIDTH - UI_SIDEBAR_WIDTH) * 0.7, HEIGHT * 0.7)
        help_rect.center = ((WIDTH - UI_SIDEBAR_WIDTH) // 2, HEIGHT // 2)
        self.help_window = pygame_gui.elements.UIWindow(
            rect=help_rect,
            manager=self.ui_manager,
            window_display_title="Help & Controls",
            visible=False,
        )
        help_text = """
<b>N-Body Simulator Controls & Info</b><br>
------------------------------------<br><br>
<b>Mouse Controls:</b><br>
 - <b>Left-Click Body:</b> Select a body to view/edit its mass.<br>
 - <b>Left-Click & Drag Body:</b> Move a selected body (resets velocity if paused).<br>
 - <b>Left-Click & Drag Background:</b> Pan the camera view.<br>
 - <b>Right-Click & Drag:</b> Add a new body at the click position. Drag direction and length set initial velocity.<br>
 - <b>Mouse Wheel Up/Down:</b> Zoom in / Zoom out.<br><br>
<b>Keyboard Shortcuts:</b><br>
 - <b>SPACE:</b> Pause / Resume simulation.<br>
 - <b>R:</b> Reset simulation to the currently selected preset.<br>
 - <b>T:</b> Toggle visibility of trails.<br>
 - <b>G:</b> Toggle gravitational field heatmap (requires Matplotlib).<br>
 - <b>B:</b> Toggle simulation boundaries (reflecting walls).<br>
 - <b>A:</b> Toggle adaptive time stepping.<br>
 - <b>C or HOME:</b> Center view on Center of Mass & Reset Zoom.<br>
 - <b>H:</b> Show / Hide this help window.<br><br>
"""
        pygame_gui.elements.UITextBox(
            html_text=help_text,
            relative_rect=pygame.Rect((0, 0), self.help_window.get_container().get_size()),
            manager=self.ui_manager,
            container=self.help_window,
        )

    def _init_state(self):
        self.bodies = []
        self.current_preset_name = "Sun & Earth"
        self.time_step = TIME_STEP_BASE
        self.paused = False
        self.running = True
        self.use_boundaries = True
        self.selected_body = None
        self.dragging_body = False
        self.dragging_camera = False
        self.camera_drag_start_screen = np.zeros(2)
        self.camera_drag_start_pan = np.zeros(2)
        self.current_zoom = ZOOM_BASE
        self.current_pan = INITIAL_PAN_OFFSET.copy()
        self.target_zoom = ZOOM_BASE
        self.target_pan = INITIAL_PAN_OFFSET.copy()
        self.simulation_time = 0.0
        self.next_body_mass = DEFAULT_NEXT_BODY_MASS
        self.next_body_radius_pixels = DEFAULT_NEXT_BODY_RADIUS_PIXELS
        self.frame_times = deque(maxlen=60)
        self.color_options = [
            EARTH_COLOR,
            MARS_COLOR,
            VENUS_COLOR,
            MERCURY_COLOR,
            GAS_COLOR,
            ICE_COLOR,
            STAR_COLORS[1],
            STAR_COLORS[2],
        ]
        self.color_index = 0
        self.adding_body_state = 0
        self.add_body_start_screen = np.zeros(2)
        self.gravity_multiplier = 10.0
        self._load_preset(self.current_preset_name)
        self.gravity_slider.set_current_value(self.gravity_multiplier)

    def _load_preset(self, preset_name):
        if preset_name not in PRESETS:
            print(f"Preset '{preset_name}' not found")
            return
        self.bodies = []
        Body.ID_counter = 0
        for config in PRESETS[preset_name]:
            mass = config.get("mass", EARTH_MASS)
            x_sim = config.get("x", 0.0)
            y_sim = config.get("y", 0.0)
            vx = config.get("vx", 0.0)
            vy = config.get("vy", 0.0)
            color = config.get("color", EARTH_COLOR)
            radius = config.get("radius", DEFAULT_NEXT_BODY_RADIUS_PIXELS)
            fixed = config.get("fixed", False)
            name = config.get("name", f"Body_{Body.ID_counter}")
            self.bodies.append(
                Body(mass, x_sim, y_sim, vx, vy, color, radius, fixed=fixed, name=name, show_trail=self.show_trails)
            )
        self.simulation_time = 0.0
        self.time_step = TIME_STEP_BASE
        self.current_preset_name = preset_name
        self.selected_body = None
        self.selected_body_name_label.set_text("Name: None")
        self.selected_body_mass_label.set_text("Mass: N/A")
        self.selected_mass_slider.disable()
        self.gravity_multiplier = 10.0
        self.gravity_slider.set_current_value(self.gravity_multiplier)
        self.gravity_label.set_text(f"Gravity: {self.gravity_multiplier:.2f}x")
        self.status_text_label.set_text(f"Preset loaded: {preset_name}")
        self.target_zoom = ZOOM_BASE
        com_pos, _ = calculate_center_of_mass(self.bodies)
        if com_pos is not None:
            screen_center = np.array([(WIDTH - UI_SIDEBAR_WIDTH) / 2, (HEIGHT - UI_BOTTOM_HEIGHT) / 2])
            self.target_pan = screen_center - com_pos * self.target_zoom
        else:
            self.target_pan = INITIAL_PAN_OFFSET.copy()
        self.current_zoom = self.target_zoom
        self.current_pan = self.target_pan.copy()

    def _process_events(self, real_dt):
        mouse_screen_pos = np.array(pygame.mouse.get_pos())
        mouse_over_control = self.control_panel_rect.collidepoint(mouse_screen_pos)
        mouse_over_status = self.status_bar_rect.collidepoint(mouse_screen_pos)
        mouse_over_ui = mouse_over_control or mouse_over_status
        mouse_world_pos = (
            (mouse_screen_pos - self.current_pan) / (self.current_zoom + 1e-18)
            if not mouse_over_ui
            else None
        )
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            self.ui_manager.process_events(event)
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.play_pause_button:
                        self.paused = not self.paused
                        self.play_pause_button.set_text("Resume" if self.paused else "Pause")
                    elif event.ui_element == self.reset_button:
                        self._load_preset(self.current_preset_name)
                    elif event.ui_element == self.trails_button:
                        self.show_trails = not self.show_trails
                        self.trails_button.set_text(f"Trails: {'ON' if self.show_trails else 'OFF'}")
                        for b in self.bodies:
                            b.show_trail = self.show_trails
                    elif event.ui_element == self.gfield_button:
                        self.show_grav_field = not self.show_grav_field
                        self.gfield_button.set_text(f"Grav Field: {'ON' if self.show_grav_field else 'OFF'}")
                    elif event.ui_element == self.boundaries_button:
                        self.use_boundaries = not self.use_boundaries
                        self.boundaries_button.set_text(f"Boundaries: {'ON' if self.use_boundaries else 'OFF'}")
                    elif event.ui_element == self.adaptive_button:
                        self.adaptive_stepping = not self.adaptive_stepping
                        self.adaptive_button.set_text(f"Adaptive Step: {'ON' if self.adaptive_stepping else 'OFF'}")
                    elif event.ui_element in {self.center_button, self.zoom_reset_button}:
                        self.target_zoom = ZOOM_BASE
                        com_pos, _ = calculate_center_of_mass(self.bodies)
                        if com_pos is not None:
                            screen_center = np.array([(WIDTH - UI_SIDEBAR_WIDTH) / 2, (HEIGHT - UI_BOTTOM_HEIGHT) / 2])
                            self.target_pan = screen_center - com_pos * self.target_zoom
                        else:
                            self.target_pan = INITIAL_PAN_OFFSET.copy()
                        self.current_zoom = self.target_zoom
                        self.current_pan = self.target_pan.copy()
                    elif event.ui_element == self.help_button:
                        self.help_window.show()
                elif event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == self.preset_dropdown:
                        self._load_preset(event.text)
                elif event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    if event.ui_element == self.speed_slider:
                        self.speed_factor = event.value
                        self.speed_label.set_text(f"Speed: {self.speed_factor:.2f}x")
                    elif event.ui_element == self.gravity_slider:
                        self.gravity_multiplier = event.value
                        self.gravity_label.set_text(f"Gravity: {self.gravity_multiplier:.2f}x")
                    elif event.ui_element == self.add_mass_slider:
                        mass_min_log = math.log10(max(1e18, 0.001 * EARTH_MASS))
                        mass_max_log = math.log10(100 * SOLAR_MASS)
                        log_mass = mass_min_log + event.value * (mass_max_log - mass_min_log)
                        self.next_body_mass = 10 ** log_mass
                        self.mass_label.set_text(f"Mass: {mass_to_display(self.next_body_mass)}")
                    elif event.ui_element == self.add_radius_slider:
                        self.next_body_radius_pixels = int(event.value)
                        self.radius_label.set_text(f"Radius: {self.next_body_radius_pixels} px")
                    elif event.ui_element == self.selected_mass_slider:
                        if self.selected_body and not self.selected_body.fixed:
                            mass_min_log = math.log10(max(1e18, 0.001 * EARTH_MASS))
                            mass_max_log = math.log10(100 * SOLAR_MASS)
                            log_mass = mass_min_log + event.value * (mass_max_log - mass_min_log)
                            self.selected_body.mass = 10 ** log_mass
                            self.selected_body_mass_label.set_text(
                                f"Mass: {mass_to_display(self.selected_body.mass)}"
                            )
            elif event.type == pygame.MOUSEBUTTONDOWN and not mouse_over_ui:
                if event.button == 1:
                    clicked = None
                    for body in reversed(self.bodies):
                        body_screen = body.get_screen_pos(self.current_zoom, self.current_pan)
                        effective_scale = max(0.1, self.current_zoom ** BODY_ZOOM_SCALING_POWER)
                        select_radius_sq = (max(5, body.radius_pixels * effective_scale)) ** 2
                        if np.sum((np.array(body_screen) - mouse_screen_pos) ** 2) < select_radius_sq:
                            clicked = body
                            break
                    if clicked:
                        self.selected_body = clicked
                        self.dragging_body = True
                        mouse_offset = mouse_screen_pos - np.array(body_screen)
                        self.status_text_label.set_text(f"Selected: {self.selected_body.name}")
                        self.selected_body_name_label.set_text(f"Name: {self.selected_body.name}")
                        self.selected_body_mass_label.set_text(
                            f"Mass: {mass_to_display(self.selected_body.mass)}"
                        )
                        if not self.selected_body.fixed:
                            mass_min_log = math.log10(max(1e18, 0.001 * EARTH_MASS))
                            mass_max_log = math.log10(100 * SOLAR_MASS)
                            cur_log_mass = math.log10(max(1e-9, self.selected_body.mass))
                            slider_val = 0.5
                            if mass_max_log > mass_min_log:
                                slider_val = (cur_log_mass - mass_min_log) / (mass_max_log - mass_min_log)
                                slider_val = max(0.0, min(1.0, slider_val))
                            self.selected_mass_slider.enable()
                            self.selected_mass_slider.set_current_value(slider_val)
                        else:
                            self.selected_mass_slider.disable()
                    else:
                        self.dragging_camera = True
                        self.camera_drag_start_screen = mouse_screen_pos.copy()
                        self.camera_drag_start_pan = self.current_pan.copy()
                elif event.button == 3:
                    self.adding_body_state = 1
                    self.add_body_start_screen = mouse_screen_pos.copy()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_body = False
                    self.dragging_camera = False
                elif event.button == 3 and self.adding_body_state == 1:
                    start_world = (self.add_body_start_screen - self.current_pan) / (self.current_zoom + 1e-18)
                    end_world = mouse_world_pos if mouse_world_pos is not None else start_world
                    vel = (end_world - start_world) * VELOCITY_DRAG_SCALE
                    color = self.color_options[self.color_index % len(self.color_options)]
                    self.color_index += 1
                    self.bodies.append(
                        Body(
                            self.next_body_mass,
                            start_world[0],
                            start_world[1],
                            vel[0],
                            vel[1],
                            color,
                            self.next_body_radius_pixels,
                            name=f"Body_{Body.ID_counter}",
                        )
                    )
                    self.adding_body_state = 0
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_body and self.selected_body:
                    new_world = (mouse_screen_pos - self.current_pan) / (self.current_zoom + 1e-18)
                    self.selected_body.pos = new_world
                    if self.paused:
                        self.selected_body.vel = np.zeros(2)
                elif self.dragging_camera:
                    delta = mouse_screen_pos - self.camera_drag_start_screen
                    self.target_pan = self.camera_drag_start_pan + delta
            elif event.type == pygame.MOUSEWHEEL and not mouse_over_ui:
                factor = ZOOM_FACTOR if event.y > 0 else 1 / ZOOM_FACTOR
                self.target_zoom *= factor
                zoom_point = mouse_screen_pos
                self.target_pan = zoom_point - (zoom_point - self.target_pan) * factor
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.play_pause_button.set_text("Resume" if self.paused else "Pause")
                elif event.key == pygame.K_r:
                    self._load_preset(self.current_preset_name)
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                    self.trails_button.set_text(f"Trails: {'ON' if self.show_trails else 'OFF'}")
                    for b in self.bodies:
                        b.show_trail = self.show_trails
                elif event.key == pygame.K_g:
                    self.show_grav_field = not self.show_grav_field
                    self.gfield_button.set_text(f"Grav Field: {'ON' if self.show_grav_field else 'OFF'}")
                elif event.key == pygame.K_b:
                    self.use_boundaries = not self.use_boundaries
                    self.boundaries_button.set_text(f"Boundaries: {'ON' if self.use_boundaries else 'OFF'}")
                elif event.key == pygame.K_a:
                    self.adaptive_stepping = not self.adaptive_stepping
                    self.adaptive_button.set_text(f"Adaptive Step: {'ON' if self.adaptive_stepping else 'OFF'}")
                elif event.key in (pygame.K_c, pygame.K_HOME):
                    self.target_zoom = ZOOM_BASE
                    com_pos, _ = calculate_center_of_mass(self.bodies)
                    if com_pos is not None:
                        screen_center = np.array([(WIDTH - UI_SIDEBAR_WIDTH) / 2, (HEIGHT - UI_BOTTOM_HEIGHT) / 2])
                        self.target_pan = screen_center - com_pos * self.target_zoom
                    else:
                        self.target_pan = INITIAL_PAN_OFFSET.copy()
                    self.current_zoom = self.target_zoom
                    self.current_pan = self.target_pan.copy()
                elif event.key == pygame.K_h:
                    if self.help_window.visible:
                        self.help_window.hide()
                    else:
                        self.help_window.show()

    def _update_physics(self, real_dt):
        self.current_zoom += (self.target_zoom - self.current_zoom) * CAMERA_SMOOTHING
        self.current_pan += (self.target_pan - self.current_pan) * CAMERA_SMOOTHING
        if not self.paused and self.bodies:
            current_g = INITIAL_G * self.gravity_multiplier
            sim_dt = self.time_step * self.speed_factor
            bounds = get_world_bounds_sim(self.current_zoom, self.current_pan) if self.use_boundaries else None
            if self.adaptive_stepping:
                advanced, next_dt = adaptive_rk4_step(
                    self.bodies, sim_dt, current_g, ERROR_TOLERANCE, self.use_boundaries, bounds
                )
                self.time_step = next_dt
                self.simulation_time += advanced
            else:
                new_pos, new_vel = perform_rk4_step(self.bodies, sim_dt, current_g)
                for i, body in enumerate(self.bodies):
                    if not body.fixed:
                        body.update_physics_state(new_pos[i], new_vel[i])
                        if self.use_boundaries and bounds is not None:
                            body.handle_boundary_collision(bounds)
                self.simulation_time += sim_dt
            remove = detect_and_handle_collisions(self.bodies, merge_on_collision=False)
            for idx in remove:
                if 0 <= idx < len(self.bodies):
                    del self.bodies[idx]
        for body in self.bodies:
            body.update_trail(self.current_zoom, self.current_pan)

    def _render(self, real_dt):
        self.screen.fill(BLACK)
        if self.show_grav_field:
            render_gravitational_field(
                self.screen,
                self.bodies,
                INITIAL_G * self.gravity_multiplier,
                self.current_zoom,
                self.current_pan,
            )
        num_drawn = 0
        for body in self.bodies:
            if num_drawn >= MAX_DISPLAY_BODIES:
                break
            show_labels = self.current_zoom > (ZOOM_BASE * 0.1)
            body.draw(self.screen, self.current_zoom, self.current_pan, draw_labels=show_labels)
            num_drawn += 1
        if self.adding_body_state == 1:
            start = self.add_body_start_screen
            end = np.array(pygame.mouse.get_pos())
            pygame.draw.aaline(self.screen, WHITE, start, end)
            pygame.gfxdraw.aacircle(
                self.screen,
                int(start[0]),
                int(start[1]),
                self.next_body_radius_pixels,
                self.color_options[self.color_index % len(self.color_options)],
            )
        if len(self.bodies) > 0:
            com_pos, _ = calculate_center_of_mass(self.bodies)
            if com_pos is not None:
                com_screen = com_pos * self.current_zoom + self.current_pan
                com_x, com_y = int(com_screen[0]), int(com_screen[1])
                if 0 <= com_x < (WIDTH - UI_SIDEBAR_WIDTH) and 0 <= com_y < HEIGHT:
                    marker_size = 4
                    pygame.draw.line(self.screen, LIGHT_GRAY, (com_x - marker_size, com_y), (com_x + marker_size, com_y), 1)
                    pygame.draw.line(self.screen, LIGHT_GRAY, (com_x, com_y - marker_size), (com_x, com_y + marker_size), 1)
        if not self.paused:
            ticks = pygame.time.get_ticks()
            if not hasattr(self, "_last_status_update") or ticks - self._last_status_update > 100:
                self._last_status_update = ticks
                info = (
                    f"Time: {time_to_display(self.simulation_time)} | "
                    f"Bodies: {len(self.bodies)} | Step: {self.time_step:.1f}s | "
                    f"Zoom: {self.current_zoom/ZOOM_BASE:.1f}x"
                )
                self.status_text_label.set_text(info)
        self.ui_manager.update(real_dt)
        self.ui_manager.draw_ui(self.screen)
        frame_duration = real_dt
        self.frame_times.append(frame_duration)
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        fps_font = pygame.font.Font(None, 18)
        fps_text = fps_font.render(f"FPS: {fps:.0f}", True, GRAY)
        self.screen.blit(fps_text, (5, 5))
        pygame.display.flip()

    def run(self):
        while self.running:
            real_dt = self.clock.tick(60) / 1000.0
            self._process_events(real_dt)
            self._update_physics(real_dt)
            self._render(real_dt)
        pygame.quit()
