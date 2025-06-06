import pygame
import math
import numpy as np
import pygame.gfxdraw

# Ensure Matplotlib is installed: pip install matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib not found. Gravitational field visualization disabled.")
    print("Install Matplotlib using: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False
    # Define dummy classes/functions if matplotlib is not available
    cm = None
    plt = None
    FigureCanvasAgg = None


import time
from collections import deque
# Ensure Pygame GUI is installed: pip install pygame_gui
try:
    import pygame_gui
    from pygame_gui.elements import UIButton, UILabel, UITextEntryLine, UIPanel, UIWindow
    from pygame_gui.elements import UIHorizontalSlider, UIDropDownMenu, UITextBox, UISelectionList
    PYGAME_GUI_AVAILABLE = True
except ImportError:
     print("Error: pygame_gui not found. Please install it: pip install pygame_gui")
     PYGAME_GUI_AVAILABLE = False
     # Exit if core UI library is missing
     exit()
from .constants import *
from .utils import mass_to_display, distance_to_display, time_to_display
from .presets import PRESETS
from .rendering import Body, render_gravitational_field
from .physics_utils import calculate_center_of_mass, perform_rk4_step, adaptive_rk4_step, detect_and_handle_collisions, get_world_bounds_sim





# --- Main Simulation Function ---
def main():
    """Main simulation loop."""
    # <<< Declare global flags modified within main's event loop >>>
    global SHOW_TRAILS, SHOW_GRAV_FIELD, ADAPTIVE_STEPPING, SPEED_FACTOR
    # <<< Access DEBUG counters for resetting >>>
    global DEBUG_DRAW_COUNT, DEBUG_COLLISION_COUNT, DEBUG_PHYSICS_COUNT

    # --- Pygame and UI Initialization ---
    pygame.init()
    if not PYGAME_GUI_AVAILABLE:
        print("Pygame GUI not found, exiting.")
        return # Exit if UI library failed to import

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"N-Body Simulation v{VERSION}")
    clock = pygame.time.Clock()
    # Load default UI theme or specify path to custom theme file
    ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT)) #, 'path/to/theme.json')

    # --- UI Element Creation ---
    # Panels
    control_panel_rect = pygame.Rect((WIDTH - UI_SIDEBAR_WIDTH, 0), (UI_SIDEBAR_WIDTH, HEIGHT))
    status_bar_rect = pygame.Rect((0, HEIGHT - UI_BOTTOM_HEIGHT), (WIDTH - UI_SIDEBAR_WIDTH, UI_BOTTOM_HEIGHT))

    control_panel = pygame_gui.elements.UIPanel(
        relative_rect=control_panel_rect,
        manager=ui_manager
    )
    status_bar = pygame_gui.elements.UIPanel(
        relative_rect=status_bar_rect,
        manager=ui_manager
    )
    # Control Panel Content (Layout using y_pos)
    y_pos = 10
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text="N-Body Simulator", manager=ui_manager, container=control_panel, object_id='#title_label')
    y_pos += 40
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (80, 30)), text="Preset:", manager=ui_manager, container=control_panel)
    preset_dropdown = pygame_gui.elements.UIDropDownMenu(options_list=list(PRESETS.keys()), starting_option="Sun & Earth", relative_rect=pygame.Rect((100, y_pos), (UI_SIDEBAR_WIDTH - 120, 30)), manager=ui_manager, container=control_panel)
    y_pos += 40
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Simulation", manager=ui_manager, container=control_panel, object_id='#section_header')
    y_pos += 25
    play_pause_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text="Pause", manager=ui_manager, container=control_panel)
    reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(((UI_SIDEBAR_WIDTH - 30) // 2 + 20, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text="Reset", manager=ui_manager, container=control_panel)
    y_pos += 40
    speed_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text=f"Speed: {SPEED_FACTOR:.1f}x", manager=ui_manager, container=control_panel)
    y_pos += 20
    speed_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), start_value=SPEED_FACTOR, value_range=(0.05, 5.0), manager=ui_manager, container=control_panel)
    y_pos += 30
    # Gravity Multiplier Slider
    gravity_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Gravity: 10.0x", manager=ui_manager, container=control_panel) # Start label at 10.0x
    y_pos += 20
    # Range increased to 100x
    gravity_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), start_value=10.0, value_range=(0.0, 100.0), manager=ui_manager, container=control_panel)
    y_pos += 30
    adaptive_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text=f"Adaptive Step: {'ON' if ADAPTIVE_STEPPING else 'OFF'}", manager=ui_manager, container=control_panel)
    y_pos += 40
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Visualization", manager=ui_manager, container=control_panel, object_id='#section_header')
    y_pos += 25
    trails_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text=f"Trails: {'ON' if SHOW_TRAILS else 'OFF'}", manager=ui_manager, container=control_panel)
    gfield_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(((UI_SIDEBAR_WIDTH - 30) // 2 + 20, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text=f"Grav Field: {'OFF'}", manager=ui_manager, container=control_panel)
    y_pos += 40
    trail_len_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
                                                 text=f"Trail Length: {DEFAULT_TRAIL_LENGTH}",
                                                 manager=ui_manager, container=control_panel)
    y_pos += 20
    trail_len_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)),
                                                             start_value=DEFAULT_TRAIL_LENGTH,
                                                             value_range=(MIN_TRAIL_LENGTH, MAX_TRAIL_LENGTH),
                                                             manager=ui_manager, container=control_panel)
    y_pos += 30
    boundaries_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text="Boundaries: ON", manager=ui_manager, container=control_panel)
    y_pos += 40
    merge_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text=f"Merge: {'ON' if MERGE_ON_COLLISION else 'OFF'}", manager=ui_manager, container=control_panel)
    y_pos += 40
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Camera", manager=ui_manager, container=control_panel, object_id='#section_header')
    y_pos += 25
    center_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text="Center View", manager=ui_manager, container=control_panel)
    zoom_reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(((UI_SIDEBAR_WIDTH - 30) // 2 + 20, y_pos), ((UI_SIDEBAR_WIDTH - 30) // 2, 30)), text="Reset Zoom", manager=ui_manager, container=control_panel)
    y_pos += 40

    # --- Add Body Controls ---
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Add Body", manager=ui_manager, container=control_panel, object_id='#section_header')
    y_pos += 25
    mass_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text=f"Mass: {mass_to_display(DEFAULT_NEXT_BODY_MASS)}", manager=ui_manager, container=control_panel)
    y_pos += 20
    mass_min_log = math.log10(max(1e18, 0.001 * EARTH_MASS))
    mass_max_log = math.log10(100 * SOLAR_MASS)
    mass_default_log = math.log10(max(1e-9, DEFAULT_NEXT_BODY_MASS))
    mass_slider_default = 0.5
    if (mass_max_log > mass_min_log):
        mass_slider_default = (mass_default_log - mass_min_log) / (mass_max_log - mass_min_log)
        mass_slider_default = max(0.0, min(1.0, mass_slider_default))
    add_mass_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), start_value=mass_slider_default, value_range=(0.0, 1.0), manager=ui_manager, container=control_panel)
    y_pos += 30
    radius_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text=f"Radius: {DEFAULT_NEXT_BODY_RADIUS_PIXELS} px", manager=ui_manager, container=control_panel)
    y_pos += 20
    add_radius_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), start_value=DEFAULT_NEXT_BODY_RADIUS_PIXELS, value_range=(1, 50), manager=ui_manager, container=control_panel)
    y_pos += 30
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 40)), text="Right-click & drag in view\nto add body with velocity.", manager=ui_manager, container=control_panel, object_id="#instruction_label")
    y_pos += 50

    # --- Selected Body Controls ---
    pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Selected Body", manager=ui_manager, container=control_panel, object_id='#section_header')
    y_pos += 25
    selected_body_name_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Name: None", manager=ui_manager, container=control_panel)
    y_pos += 25
    selected_body_mass_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), text="Mass: N/A", manager=ui_manager, container=control_panel)
    y_pos += 20
    selected_mass_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 20)), start_value=0.5, value_range=(0.0, 1.0), manager=ui_manager, container=control_panel)
    selected_mass_slider.disable() # Disabled by default
    y_pos += 40
    # --- END Selected Body Controls ---

    help_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text="Help / Controls", manager=ui_manager, container=control_panel)
    # Status Bar Text
    status_text_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 0), (WIDTH - UI_SIDEBAR_WIDTH - 20, UI_BOTTOM_HEIGHT)), text="Status: Initializing...", manager=ui_manager, container=status_bar)
    # Help Window
    help_window_rect = pygame.Rect(0, 0, (WIDTH - UI_SIDEBAR_WIDTH) * 0.7, HEIGHT * 0.7)
    help_window_rect.center = ((WIDTH - UI_SIDEBAR_WIDTH) // 2, HEIGHT // 2)
    help_window = pygame_gui.elements.UIWindow(rect=help_window_rect, manager=ui_manager, window_display_title="Help & Controls", visible=False)
    help_text_content = """
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
 - <b>Delete:</b> Remove the currently selected body.<br><br>

 <b>UI Panel:</b><br>
 - Use sliders and buttons to control parameters, add bodies, edit mass, adjust trail length, and toggle collision merging.<br><br>

<b>Physics Notes:</b><br>
 - Uses RK4 integration (fixed or adaptive step).<br>
 - Gravitational forces calculated using Newton's law with softening for close encounters.<br>
 - Collisions are detected based on calculated physical radii (mass-dependent) and result in bounces (can be changed to merge).<br>
 - Energy/Momentum calculations are approximate due to numerical integration and softening.
    """ # Same help text as before
    help_textbox = pygame_gui.elements.UITextBox(html_text=help_text_content, relative_rect=pygame.Rect((0, 0), help_window.get_container().get_size()), manager=ui_manager, container=help_window)


    # --- Simulation State Variables (defined within main) ---
    bodies = []
    current_preset_name = "Sun & Earth" # Initial preset
    time_step = TIME_STEP_BASE # Current adaptive step size suggestion
    paused = False
    running = True
    use_boundaries = True
    merge_on_collision = MERGE_ON_COLLISION
    selected_body = None # Body currently being dragged OR selected for editing
    dragging_body = False # Flag specifically for dragging motion
    dragging_camera = False
    camera_drag_start_screen = np.zeros(2)
    camera_drag_start_pan = np.zeros(2)
    # --- Initialize camera variables INSIDE main ---
    current_zoom = ZOOM_BASE
    current_pan = INITIAL_PAN_OFFSET.copy()
    target_zoom = ZOOM_BASE # Target for smooth zoom
    target_pan = INITIAL_PAN_OFFSET.copy() # Target for smooth pan
    # --- End camera variables ---
    simulation_time = 0.0 # Total simulated time elapsed (seconds)
    next_body_mass = DEFAULT_NEXT_BODY_MASS
    next_body_radius_pixels = DEFAULT_NEXT_BODY_RADIUS_PIXELS
    trail_length = DEFAULT_TRAIL_LENGTH
    frame_times = deque(maxlen=60) # For FPS calculation
    color_options = [EARTH_COLOR, MARS_COLOR, VENUS_COLOR, MERCURY_COLOR, GAS_COLOR, ICE_COLOR, STAR_COLORS[1], STAR_COLORS[2]]
    color_index = 0
    adding_body_state = 0 # 0: idle, 1: dragging to set velocity
    add_body_start_screen = np.zeros(2) # Screen position where right-click started
    # <<< Gravity multiplier state - Default set to 10.0 >>>
    gravity_multiplier = 10.0
    # --- End Simulation State Variables ---


    # --- Helper Function to Load Preset ---
    def load_preset(preset_name):
        # Use nonlocal to modify variables in the outer 'main' scope
        nonlocal bodies, simulation_time, time_step, current_preset_name, selected_body
        # Use nonlocal for camera variables as well, since they are now defined in main
        nonlocal target_zoom, target_pan, current_zoom, current_pan
        # <<< Use nonlocal for gravity_multiplier >>>
        nonlocal gravity_multiplier, trail_length
        # Use global for module-level DEBUG counters
        global DEBUG_DRAW_COUNT, DEBUG_COLLISION_COUNT, DEBUG_PHYSICS_COUNT
        DEBUG_DRAW_COUNT = 0 # Reset debug counters
        DEBUG_COLLISION_COUNT = 0
        DEBUG_PHYSICS_COUNT = 0


        if preset_name not in PRESETS:
            print(f"Error: Preset '{preset_name}' not found.")
            return

        preset_data = PRESETS[preset_name]
        bodies = []
        Body.ID_counter = 0 # Reset unique IDs
        for body_config in preset_data:
            # Ensure all required keys are present with defaults
            mass = body_config.get("mass", EARTH_MASS)
            x_sim = body_config.get("x", 0.0) # Assume preset 'x' is sim units
            y_sim = body_config.get("y", 0.0) # Assume preset 'y' is sim units
            vx_m_s = body_config.get("vx", 0.0)
            vy_m_s = body_config.get("vy", 0.0)
            color = body_config.get("color", EARTH_COLOR)
            radius_px = body_config.get("radius", DEFAULT_NEXT_BODY_RADIUS_PIXELS)
            fixed = body_config.get("fixed", False)
            name = body_config.get("name", f"Body_{Body.ID_counter}")

            body = Body(mass, x_sim, y_sim, vx_m_s, vy_m_s, color, radius_px,
                         fixed=fixed, name=name, show_trail=SHOW_TRAILS)
            body.set_trail_length(trail_length)
            bodies.append(body)

        simulation_time = 0.0
        time_step = TIME_STEP_BASE # Reset adaptive step size
        current_preset_name = preset_name
        selected_body = None # Deselect body on preset load
        selected_body_name_label.set_text("Name: None")
        selected_body_mass_label.set_text("Mass: N/A")
        selected_mass_slider.disable()
        # <<< Reset gravity multiplier and slider on preset load to new default >>>
        gravity_multiplier = 10.0
        gravity_slider.set_current_value(gravity_multiplier)
        gravity_label.set_text(f"Gravity: {gravity_multiplier:.2f}x")
        trail_length = DEFAULT_TRAIL_LENGTH
        trail_len_slider.set_current_value(trail_length)
        trail_len_label.set_text(f"Trail Length: {trail_length}")
        # <<< End Reset >>>
        status_text_label.set_text(f"Preset loaded: {preset_name}")
        # Reset camera to view the new preset
        target_zoom = ZOOM_BASE
        com_pos, _ = calculate_center_of_mass(bodies)
        if com_pos is not None:
             screen_center = np.array([(WIDTH - UI_SIDEBAR_WIDTH) / 2, (HEIGHT - UI_BOTTOM_HEIGHT) / 2])
             target_pan = screen_center - com_pos * target_zoom
        else:
             target_pan = INITIAL_PAN_OFFSET.copy()
        # Instantly update current camera for immediate view change
        current_zoom = target_zoom
        current_pan = target_pan.copy()


    # Load the initial preset
    load_preset(current_preset_name)
    # Set initial gravity slider value
    gravity_slider.set_current_value(gravity_multiplier)
    trail_len_slider.set_current_value(trail_length)


    # --- Main Game Loop ---
    while running:
        real_dt = clock.tick(60) / 1000.0  # Real time elapsed (seconds), capped at 60 FPS
        frame_start_time = time.time()

        # --- Input Handling ---
        mouse_screen_pos = np.array(pygame.mouse.get_pos())
        # Check if mouse is over the simulation area (not sidebar or statusbar)
        mouse_over_control_panel = control_panel_rect.collidepoint(mouse_screen_pos)
        mouse_over_status_bar = status_bar_rect.collidepoint(mouse_screen_pos)
        mouse_over_ui = mouse_over_control_panel or mouse_over_status_bar

        # Calculate mouse position in simulation world coordinates (if not over UI)
        # Add small epsilon to zoom to prevent division by zero if zoom becomes extremely small
        mouse_world_pos = (mouse_screen_pos - current_pan) / (current_zoom + 1e-18) if not mouse_over_ui else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Pass event to UI Manager FIRST
            ui_manager.process_events(event)

            # --- Handle UI Events ---
            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    # --- Button Actions ---
                    if event.ui_element == play_pause_button:
                        paused = not paused
                        play_pause_button.set_text("Resume" if paused else "Pause")
                    elif event.ui_element == reset_button:
                        load_preset(current_preset_name) # Reload current preset
                    elif event.ui_element == trails_button:
                        # Modify module-level variable directly (global already declared at top of main)
                        SHOW_TRAILS = not SHOW_TRAILS
                        trails_button.set_text(f"Trails: {'ON' if SHOW_TRAILS else 'OFF'}")
                        for body in bodies: body.show_trail = SHOW_TRAILS
                    elif event.ui_element == gfield_button:
                        # Modify module-level variable directly
                        SHOW_GRAV_FIELD = not SHOW_GRAV_FIELD
                        gfield_button.set_text(f"Grav Field: {'ON' if SHOW_GRAV_FIELD else 'OFF'}")
                    elif event.ui_element == boundaries_button:
                        use_boundaries = not use_boundaries
                        boundaries_button.set_text(f"Boundaries: {'ON' if use_boundaries else 'OFF'}")
                    elif event.ui_element == merge_button:
                        merge_on_collision = not merge_on_collision
                        merge_button.set_text(f"Merge: {'ON' if merge_on_collision else 'OFF'}")
                    elif event.ui_element == adaptive_button:
                        # Modify module-level variable directly
                        ADAPTIVE_STEPPING = not ADAPTIVE_STEPPING
                        adaptive_button.set_text(f"Adaptive Step: {'ON' if ADAPTIVE_STEPPING else 'OFF'}")
                    elif event.ui_element == center_button or event.ui_element == zoom_reset_button:
                        # Reset zoom and center on COM
                        target_zoom = ZOOM_BASE
                        com_pos, _ = calculate_center_of_mass(bodies)
                        if com_pos is not None:
                             screen_center = np.array([(WIDTH - UI_SIDEBAR_WIDTH) / 2, (HEIGHT - UI_BOTTOM_HEIGHT) / 2])
                             target_pan = screen_center - com_pos * target_zoom
                        else:
                             target_pan = INITIAL_PAN_OFFSET.copy() # Reset pan if no bodies
                        # Instantly update current camera as well for immediate reset feel
                        current_zoom = target_zoom
                        current_pan = target_pan.copy()
                    elif event.ui_element == help_button:
                         help_window.show()

                elif event.user_type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                    if event.ui_element == preset_dropdown:
                        load_preset(event.text) # Load selected preset

                elif event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                    # --- Slider Actions ---
                    if event.ui_element == speed_slider:
                         # Modify module-level variable directly
                         SPEED_FACTOR = event.value
                         speed_label.set_text(f"Speed: {SPEED_FACTOR:.2f}x")
                    elif event.ui_element == gravity_slider:
                         gravity_multiplier = event.value
                         gravity_label.set_text(f"Gravity: {gravity_multiplier:.2f}x")
                    elif event.ui_element == add_mass_slider: # Renamed slider
                        log_mass = mass_min_log + event.value * (mass_max_log - mass_min_log)
                        next_body_mass = 10**log_mass
                        mass_label.set_text(f"Mass: {mass_to_display(next_body_mass)}")
                    elif event.ui_element == add_radius_slider: # Renamed slider
                        next_body_radius_pixels = int(event.value)
                        radius_label.set_text(f"Radius: {next_body_radius_pixels} px")
                    elif event.ui_element == selected_mass_slider: # <<< Handle selected body mass slider >>>
                        if selected_body and not selected_body.fixed:
                            # Convert slider value (0-1) back to mass using log scale
                            log_mass = mass_min_log + event.value * (mass_max_log - mass_min_log)
                            new_mass = 10**log_mass
                            selected_body.mass = new_mass
                            # Update the label
                            selected_body_mass_label.set_text(f"Mass: {mass_to_display(selected_body.mass)}")
                    elif event.ui_element == trail_len_slider:
                        trail_length = int(event.value)
                        trail_len_label.set_text(f"Trail Length: {trail_length}")
                        for b in bodies:
                            b.set_trail_length(trail_length)


            # --- Handle Mouse & Keyboard Input (if not over UI panel for relevant events) ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                 # Check if click is outside UI panels
                 if not mouse_over_ui:
                    if event.button == 1:  # Left click
                        # Check if clicking on a body
                        clicked_on_body = None
                        for body in reversed(bodies): # Check topmost first
                            body_screen_pos = body.get_screen_pos(current_zoom, current_pan)
                            # Selection uses visual radius scaled by zoom
                            effective_zoom_scale = max(0.1, current_zoom**BODY_ZOOM_SCALING_POWER)
                            select_radius_sq = (max(5, body.radius_pixels * effective_zoom_scale))**2 # Use scaled radius
                            if np.sum((np.array(body_screen_pos) - mouse_screen_pos)**2) < select_radius_sq:
                                clicked_on_body = body
                                break # Found body

                        if clicked_on_body:
                            selected_body = clicked_on_body # Set selected body
                            dragging_body = True # Start dragging motion
                            # Offset from body center to mouse click (screen pixels)
                            mouse_offset = mouse_screen_pos - np.array(body_screen_pos)
                            status_text_label.set_text(f"Selected: {selected_body.name}")
                            # Update selected body UI
                            selected_body_name_label.set_text(f"Name: {selected_body.name}")
                            selected_body_mass_label.set_text(f"Mass: {mass_to_display(selected_body.mass)}")
                            if not selected_body.fixed:
                                # Set slider value based on current mass (log scale)
                                current_log_mass = math.log10(max(1e-9, selected_body.mass))
                                slider_val = 0.5 # Default
                                if (mass_max_log > mass_min_log):
                                     slider_val = (current_log_mass - mass_min_log) / (mass_max_log - mass_min_log)
                                     slider_val = max(0.0, min(1.0, slider_val)) # Clamp
                                selected_mass_slider.set_current_value(slider_val)
                                selected_mass_slider.enable()
                            else:
                                selected_mass_slider.disable() # Cannot edit fixed body mass

                        else: # Clicked background
                            selected_body = None # Deselect
                            dragging_body = False
                            dragging_camera = True # Start panning
                            camera_drag_start_screen = mouse_screen_pos.copy()
                            camera_drag_start_pan = current_pan.copy() # Store pan at start of drag
                            # Update selected body UI
                            selected_body_name_label.set_text("Name: None")
                            selected_body_mass_label.set_text("Mass: N/A")
                            selected_mass_slider.disable()


                    elif event.button == 3:  # Right click - Start adding body
                        if adding_body_state == 0:
                            add_body_start_screen = mouse_screen_pos.copy()
                            adding_body_state = 1
                            status_text_label.set_text("Drag to set velocity...")
                 # Handle mouse wheel zoom regardless of UI hover? Or check event consumption?
                 # Let's allow zoom even if hovering UI for now.
                 if event.button == 4:  # Mouse wheel up - Zoom in
                     if mouse_world_pos is not None: # Need a world pos to zoom towards
                         target_zoom *= ZOOM_FACTOR
                         # Adjust pan to keep mouse pointer location static in world space
                         target_pan = mouse_screen_pos - mouse_world_pos * target_zoom

                 elif event.button == 5:  # Mouse wheel down - Zoom out
                      if mouse_world_pos is not None: # Need a world pos to zoom towards
                         target_zoom /= ZOOM_FACTOR
                         target_zoom = max(1e-18, target_zoom) # Prevent zoom <= 0
                         target_pan = mouse_screen_pos - mouse_world_pos * target_zoom

            elif event.type == pygame.MOUSEBUTTONUP:
                 # Handle mouse up regardless of UI hover for releasing drags/adds
                 if event.button == 1:  # Left click release
                     # Only clear status if we were dragging a body
                     if dragging_body: status_text_label.set_text(f"Released {selected_body.name if selected_body else ''}")
                     # Keep body selected, but stop dragging motion
                     dragging_body = False
                     dragging_camera = False # Stop panning too
                 elif event.button == 3: # Right click release - Finish adding body
                     # Check if we were in the adding state
                     if adding_body_state == 1:
                         # Check if mouse is currently over UI - if so, cancel add
                         if mouse_over_ui:
                              print("Add body cancelled - mouse released over UI.")
                         elif mouse_world_pos is not None: # Ensure mouse is still valid
                             # Start position in world coords
                             add_start_world = (add_body_start_screen - current_pan) / (current_zoom + 1e-18)
                             # End position (current mouse) in world coords
                             add_end_world = mouse_world_pos

                             # Calculate velocity from drag vector (world units -> m/s)
                             drag_vector_world = add_end_world - add_start_world # Sim units
                             vel_m_s = drag_vector_world * VELOCITY_DRAG_SCALE

                             # Create new body
                            new_body = Body(
                                mass=next_body_mass,
                                x=add_start_world[0], y=add_start_world[1], # Sim units
                                vx=vel_m_s[0], vy=vel_m_s[1], # m/s
                                color=color_options[color_index],
                                radius=next_body_radius_pixels, # Pixels
                                name=f"Body_{Body.ID_counter}", # ID increments in Body init
                                show_trail=SHOW_TRAILS
                            )
                            new_body.set_trail_length(trail_length)
                            bodies.append(new_body)
                             status_text_label.set_text(f"Added {new_body.name}")
                             color_index = (color_index + 1) % len(color_options)
                         adding_body_state = 0 # Reset state regardless

            elif event.type == pygame.MOUSEMOTION:
                 # Handle motion only if not over UI? Or allow dragging over UI?
                 # Let's allow dragging over UI for now, but panning stops if mouse enters UI.
                 if dragging_body and selected_body: # Dragging selected body
                     new_body_center_screen = mouse_screen_pos - mouse_offset
                     selected_body.pos = (new_body_center_screen - current_pan) / (current_zoom + 1e-18)
                     if paused: selected_body.vel = np.zeros(2) # Reset velocity if paused
                     selected_body.clear_trail() # Avoid trail jumps while dragging

                 elif dragging_camera and not mouse_over_ui: # Panning camera only if mouse outside UI
                     drag_delta_screen = mouse_screen_pos - camera_drag_start_screen
                     # Target pan is the pan at drag start + the screen delta
                     target_pan = camera_drag_start_pan + drag_delta_screen

            elif event.type == pygame.KEYDOWN:
                 # Keyboard shortcuts usually don't need UI hover check
                 # --- Keyboard Shortcuts ---
                 if event.key == pygame.K_SPACE:
                     paused = not paused
                     play_pause_button.set_text("Resume" if paused else "Pause")
                 elif event.key == pygame.K_r:
                     load_preset(current_preset_name)
                 elif event.key == pygame.K_t:
                     # Modify module-level variable directly
                     SHOW_TRAILS = not SHOW_TRAILS
                     trails_button.set_text(f"Trails: {'ON' if SHOW_TRAILS else 'OFF'}")
                     for body in bodies: body.show_trail = SHOW_TRAILS
                 elif event.key == pygame.K_g:
                     # Modify module-level variable directly
                     SHOW_GRAV_FIELD = not SHOW_GRAV_FIELD
                     gfield_button.set_text(f"Grav Field: {'ON' if SHOW_GRAV_FIELD else 'OFF'}")
                 elif event.key == pygame.K_b:
                     use_boundaries = not use_boundaries
                     boundaries_button.set_text(f"Boundaries: {'ON' if use_boundaries else 'OFF'}")
                 elif event.key == pygame.K_a:
                      # Modify module-level variable directly
                      ADAPTIVE_STEPPING = not ADAPTIVE_STEPPING
                      adaptive_button.set_text(f"Adaptive Step: {'ON' if ADAPTIVE_STEPPING else 'OFF'}")
                elif event.key == pygame.K_c or event.key == pygame.K_HOME:
                      target_zoom = ZOOM_BASE
                      com_pos, _ = calculate_center_of_mass(bodies)
                      if com_pos is not None:
                          screen_center = np.array([(WIDTH - UI_SIDEBAR_WIDTH) / 2, (HEIGHT - UI_BOTTOM_HEIGHT) / 2])
                          target_pan = screen_center - com_pos * target_zoom
                      else: target_pan = INITIAL_PAN_OFFSET.copy()
                      # Instantly update current camera as well
                      current_zoom = target_zoom
                      current_pan = target_pan.copy()
                elif event.key == pygame.K_h:
                      if help_window.visible: help_window.hide()
                      else: help_window.show()
                elif event.key == pygame.K_DELETE and selected_body:
                     bodies.remove(selected_body)
                     selected_body = None
                     selected_body_name_label.set_text("Name: None")
                     selected_body_mass_label.set_text("Mass: N/A")
                     selected_mass_slider.disable()


        # --- Smooth Camera Motion ---
        current_zoom += (target_zoom - current_zoom) * CAMERA_SMOOTHING
        current_pan += (target_pan - current_pan) * CAMERA_SMOOTHING


        # --- Physics Update ---
        if not paused and bodies:
            # <<< Use gravity multiplier >>>
            current_g = INITIAL_G * gravity_multiplier
            # <<< DEBUG PRINT for Gravity >>>
            # if DEBUG_PHYSICS_COUNT < 5:
            #     print(f"Frame {DEBUG_PHYSICS_COUNT}: Using G = {current_g:.3e} (Multiplier: {gravity_multiplier:.2f})")
            #     if DEBUG_PHYSICS_COUNT == 4: print("--- End G Debug ---")
            #     DEBUG_PHYSICS_COUNT += 1
            # <<< End Debug Print >>>

            sim_dt_propose = time_step * SPEED_FACTOR # Proposed step size for this frame
            time_advanced_this_frame = 0.0

            # Get world bounds for boundary checks (in simulation units)
            world_bounds_sim = get_world_bounds_sim(current_zoom, current_pan) if use_boundaries else None

            if ADAPTIVE_STEPPING:
                # Adaptive step updates bodies internally if accepted
                time_advanced, next_dt_suggestion = adaptive_rk4_step(
                    bodies, sim_dt_propose, current_g, ERROR_TOLERANCE, use_boundaries, world_bounds_sim
                )
                time_step = next_dt_suggestion # Use suggestion for next frame's proposal
                time_advanced_this_frame = time_advanced # Record how much time actually passed
            else:
                # Fixed step RK4
                new_positions, new_velocities = perform_rk4_step(bodies, sim_dt_propose, current_g)
                # Update bodies manually
                for i, body in enumerate(bodies):
                    if not body.fixed:
                        body.update_physics_state(new_positions[i], new_velocities[i])
                        # Apply boundaries after update
                        if use_boundaries and world_bounds_sim is not None:
                            body.handle_boundary_collision(world_bounds_sim)
                time_advanced_this_frame = sim_dt_propose # Fixed step always advances by proposed dt

            simulation_time += time_advanced_this_frame

            # --- Handle Collisions (after integration) ---
            indices_to_remove = detect_and_handle_collisions(bodies, merge_on_collision=merge_on_collision)
            if indices_to_remove:
                 # Remove merged bodies safely (iterate backwards)
                 for index in indices_to_remove:
                     # Check index validity before deleting
                     if 0 <= index < len(bodies):
                         del bodies[index]
                     else:
                          print(f"Warning: Invalid index {index} during body removal.")


        # --- Update Trails (after physics and collisions) ---
        for body in bodies:
            body.update_trail(current_zoom, current_pan)


        # --- Drawing ---
        screen.fill(BLACK) # Clear screen

        # Draw Gravitational Field (if enabled)
        if SHOW_GRAV_FIELD:
            # Pass current_g (potentially scaled) to field renderer
            render_gravitational_field(screen, bodies, INITIAL_G * gravity_multiplier, current_zoom, current_pan)

        # Draw Bodies and Trails
        num_drawn = 0
        for body in bodies:
            if num_drawn >= MAX_DISPLAY_BODIES: break # Performance limit
            # Determine if labels should be drawn based on zoom
            show_labels = current_zoom > (ZOOM_BASE * 0.1) # Show labels if zoomed in moderately
            body.draw(screen, current_zoom, current_pan, draw_labels=show_labels)
            num_drawn += 1


        # Draw velocity preview line when adding body
        if adding_body_state == 1: # Draw preview even if mouse is over UI temporarily
            start_screen = add_body_start_screen
            end_screen = mouse_screen_pos
            # Draw line
            pygame.draw.aaline(screen, WHITE, (int(start_screen[0]), int(start_screen[1])), (int(end_screen[0]), int(end_screen[1])))
            # Draw preview circle
            pygame.gfxdraw.aacircle(screen, int(start_screen[0]), int(start_screen[1]), next_body_radius_pixels, color_options[color_index])
            # Display velocity estimate
            try: # Add try-except for font rendering
                vel_font = pygame.font.Font(None, 16)
                add_start_world = (add_body_start_screen - current_pan) / (current_zoom + 1e-18)
                # Use last known valid mouse_world_pos if current is None (over UI)
                current_mouse_world = (mouse_screen_pos - current_pan) / (current_zoom + 1e-18)
                vel_m_s = (current_mouse_world - add_start_world) * VELOCITY_DRAG_SCALE
                speed_m_s = np.linalg.norm(vel_m_s)
                vel_text = vel_font.render(f"{speed_m_s:.1f} m/s", True, WHITE)
                screen.blit(vel_text, (int(end_screen[0]) + 10, int(end_screen[1]) - 10))
            except Exception as e:
                 print(f"Velocity text error: {e}") # Debug


        # Draw Center of Mass marker (optional)
        if len(bodies) > 0:
            com_pos, _ = calculate_center_of_mass(bodies)
            if com_pos is not None:
                com_screen_pos = com_pos * current_zoom + current_pan
                com_x, com_y = int(com_screen_pos[0]), int(com_screen_pos[1])
                # Draw small cross marker if within view
                if 0 <= com_x < (WIDTH - UI_SIDEBAR_WIDTH) and 0 <= com_y < HEIGHT:
                    marker_size = 4
                    pygame.draw.line(screen, LIGHT_GRAY, (com_x - marker_size, com_y), (com_x + marker_size, com_y), 1)
                    pygame.draw.line(screen, LIGHT_GRAY, (com_x, com_y - marker_size), (com_x, com_y + marker_size), 1)


        # --- Update Status Bar ---
        if not paused:
             # Update status text less frequently for performance
             current_ticks = pygame.time.get_ticks()
             if not hasattr(main, 'last_status_update') or current_ticks - main.last_status_update > 100: # Update every 100ms
                 main.last_status_update = current_ticks
                 status_info = (f"Time: {time_to_display(simulation_time)} | "
                               f"Bodies: {len(bodies)} | Step: {time_step:.1f}s | "
                               f"Zoom: {current_zoom/ZOOM_BASE:.1f}x")
                 # Add energy calculation if needed (can be slow)
                 # try:
                 #    ke, pe, te = calculate_system_energies(bodies, INITIAL_G * gravity_multiplier) # Use scaled G
                 #    status_info += f" | E: {te:.2e} J"
                 # except Exception: pass # Ignore energy calc errors
                 status_text_label.set_text(status_info)


        # --- UI Update & Draw ---
        ui_manager.update(real_dt)
        ui_manager.draw_ui(screen) # Draw UI elements on top


        # --- FPS Counter ---
        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        frame_times.append(frame_duration)
        avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        try: # Add try-except for font rendering
            fps_font = pygame.font.Font(None, 18)
            fps_text = fps_font.render(f"FPS: {fps:.0f}", True, GRAY)
            # Position FPS counter top-left within sim area
            screen.blit(fps_text, (5, 5))
        except Exception as e:
            print(f"FPS text error: {e}") # Debug


        # --- Update Display ---
        pygame.display.flip()

    # --- End Main Loop ---
    pygame.quit()


# --- Entry Point ---
if __name__ == "__main__":
    # Check for dependencies before running main
    if not PYGAME_GUI_AVAILABLE:
         print("\nExiting due to missing pygame_gui dependency.")
    else:
        try:
            main()
        except Exception as e:
            print(f"\n--- Simulation Runtime Error ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            import traceback
            print("\n--- Traceback ---")
            traceback.print_exc()
            print("\n--------------------\n")
            pygame.quit()
            # Keep console open to see error in some environments
            input("Press Enter to exit...")

