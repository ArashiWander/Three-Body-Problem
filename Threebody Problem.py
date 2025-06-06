import pygame
import math
import random
import numpy as np
import pygame.gfxdraw
# Ensure Numba is installed: pip install numba
try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    print("Warning: Numba not found. JIT compilation disabled. Simulation might be slow.")
    print("Install Numba using: pip install numba")
    # Define dummy njit decorator if numba is not available
    def nb_njit(func):
        return func
    nb = type('obj', (object,), {'njit': nb_njit})() # Mock numba object with njit
    NUMBA_AVAILABLE = False

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
import os
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


# Application Constants
VERSION = "2.0.17" # Updated version (Aggressive Gravity/Zoom)
WIDTH, HEIGHT = 1280, 800  # Screen resolution
UI_SIDEBAR_WIDTH = 300     # Width of the control panel
UI_BOTTOM_HEIGHT = 30      # Height of the status bar

# Appearance
BLACK = (0, 0, 0)
DARK_GRAY = (20, 20, 20)
GRAY = (80, 80, 80)
LIGHT_GRAY = (160, 160, 160)
WHITE = (255, 255, 255)

# Planet/Star Colors
SUN_COLOR = (255, 230, 110)
MERCURY_COLOR = (200, 190, 190)
VENUS_COLOR = (255, 190, 150)
EARTH_COLOR = (100, 140, 255)
MARS_COLOR = (220, 100, 60)
JUPITER_COLOR = (255, 165, 60)
GAS_COLOR = (230, 230, 210)
ICE_COLOR = (160, 210, 255)
STAR_COLORS = [
    (255, 230, 180),  # F-type (yellowish white)
    (255, 210, 130),  # G-type (yellow)
    (255, 175, 100),  # K-type (orange)
    (255, 120, 80)    # M-type (red)
]

# UI Theme colors (Used if creating a custom theme file)
UI_BG_COLOR = (30, 30, 40)
UI_ACCENT_COLOR = (60, 100, 180)
UI_TEXT_COLOR = (230, 230, 230)

# Physical constants - Using SI units internally where possible
G_REAL = 6.67430e-11  # Real gravitational constant (m^3 kg^-1 s^-2)
AU = 1.496e11         # Astronomical Unit in meters
SOLAR_MASS = 1.989e30 # Mass of the Sun in kg
EARTH_MASS = 5.972e24 # Mass of the Earth in kg
JUPITER_MASS = 1.898e27 # Mass of Jupiter in kg
EARTH_RADIUS_METERS = 6.371e6 # Earth radius in meters

# Simulation scaling and parameters
# SPACE_SCALE defines how many meters one simulation unit represents.
# This is crucial for converting between physics calculations (meters) and simulation space.
SPACE_SCALE = 5e9      # Meters per simulation unit (e.g., 1 unit = 5 billion meters)
INITIAL_G = G_REAL     # Use the real gravitational constant for physics calculations
TIME_STEP_BASE = 600   # Default base timestep (in seconds) - 10 minutes
SPEED_FACTOR = 1.0     # Simulation speed multiplier (scales TIME_STEP_BASE) - MODIFIED IN MAIN
# Softening factor to prevent extreme forces during close encounters (in meters squared)
# <<< Drastically reduced softening factor >>>
SOFTENING_FACTOR_SQ = (1.0)**2 # Softening distance ~1m (m^2)
# Scale factor for converting mouse drag (in simulation units) to initial velocity (m/s)
VELOCITY_DRAG_SCALE = 10000.0 # Higher value means smaller drag gives higher velocity

# Adaptive simulation parameters
ADAPTIVE_STEPPING = True     # Enable/disable adaptive time stepping - MODIFIED IN MAIN
ERROR_TOLERANCE = 1e-6      # Target relative error for adaptive stepping (adjust for accuracy vs speed)
MIN_TIME_STEP = 10          # Minimum allowed time step (seconds)
MAX_TIME_STEP = 3600        # Maximum allowed time step (seconds, 1 hour)

# Visualization parameters
FIELD_RESOLUTION = 30     # Grid size for gravitational field visualization (lower for performance)
SHOW_GRAV_FIELD = False   # Whether to display the gravitational field by default - MODIFIED IN MAIN
SHOW_TRAILS = True        # Whether to display trails by default - MODIFIED IN MAIN
# ZOOM_BASE defines the initial view scale: simulation units per pixel.
# Smaller value means more zoomed in initially.
ZOOM_BASE = 1.0 / (AU / 500) # Initial view: ~500 pixels represent 1 AU
INITIAL_PAN_OFFSET = np.array([(WIDTH - UI_SIDEBAR_WIDTH)/2, (HEIGHT - UI_BOTTOM_HEIGHT)/2], dtype=np.float64) # Initial view center
CAMERA_SMOOTHING = 0.05   # Camera transition smoothness (0-1, lower is faster)
ZOOM_FACTOR = 1.2         # Multiplier for zoom steps
# <<< More aggressive visual zoom scaling >>>
BODY_ZOOM_SCALING_POWER = 0.2 # Power for non-linear visual scaling (1.0 = linear)

# Simulation defaults
DEFAULT_NEXT_BODY_MASS = EARTH_MASS
DEFAULT_NEXT_BODY_RADIUS_PIXELS = 8 # Default visual radius in pixels
DEFAULT_TRAIL_LENGTH = 500 # Max points in trail
MAX_DISPLAY_BODIES = 100   # Limit bodies drawn for performance (doesn't affect simulation)
# Collision distance factor
COLLISION_DISTANCE_FACTOR = 5.0 # Multiplies physical radius for collision check

# Debugging flags
DEBUG_PHYSICS_COUNT = 0 # Counter for physics debug prints


# --- Scientific Unit Conversion Helpers ---
def mass_to_display(mass_kg):
    """Convert mass in kg to display string with appropriate units (M☉, M⊕, etc.)."""
    if mass_kg == 0: return "0 kg"
    if mass_kg >= 0.1 * SOLAR_MASS:
        return f"{mass_kg/SOLAR_MASS:.2f} M☉" # Solar masses
    elif mass_kg >= 0.1 * EARTH_MASS:
        return f"{mass_kg/EARTH_MASS:.2f} M⊕" # Earth masses
    else:
        # Use scientific notation for smaller masses
        return f"{mass_kg:.2e} kg"

def distance_to_display(dist_meters):
    """Convert distance in meters to display string with appropriate units (AU, km, m)."""
    if dist_meters == 0: return "0 m"
    if abs(dist_meters) >= 0.1 * AU:
        return f"{dist_meters/AU:.2f} AU"
    elif abs(dist_meters) >= 1e6: # Megameters
        return f"{dist_meters/1e6:.2f} Mm"
    elif abs(dist_meters) >= 1e3: # Kilometers
        return f"{dist_meters/1e3:.2f} km"
    else:
        return f"{dist_meters:.1f} m"

def time_to_display(seconds):
    """Convert time in seconds to display string (years, days, hrs, min, sec)."""
    if seconds < 0: return "N/A"
    if seconds == 0: return "0 sec"

    years = seconds / 31536000 # Approx seconds in a year
    if years >= 1:
        return f"{years:.1f} years"
    days = seconds / 86400
    if days >= 1:
        return f"{days:.1f} days"
    hours = seconds / 3600
    if hours >= 1:
        return f"{hours:.1f} hrs"
    minutes = seconds / 60
    if minutes >= 1:
        return f"{minutes:.1f} min"
    else:
        return f"{seconds:.1f} sec"

# --- Celestial Presets ---
# Define presets using physical units (mass in kg, positions relative to center in simulation units, velocity in m/s)
# Position Note: (0,0) in simulation units corresponds to the center of the initial view.
PRESETS = {
    "Empty": [],
    "Sun & Earth": [
        {"mass": SOLAR_MASS, "x": 0, "y": 0, "vx": 0, "vy": 0, "color": SUN_COLOR, "radius": 15, "name": "Sun", "fixed": True},
        {"mass": EARTH_MASS, "x": AU / SPACE_SCALE, "y": 0, "vx": 0, "vy": 29780, "color": EARTH_COLOR, "radius": 8, "name": "Earth"} # Earth orbiting Sun
    ],
    "Earth & Moon": [
        {"mass": EARTH_MASS, "x": 0, "y": 0, "vx": 0, "vy": 0, "color": EARTH_COLOR, "radius": 12, "name": "Earth", "fixed": True},
        {"mass": 7.342e22, "x": 384400e3 / SPACE_SCALE, "y": 0, "vx": 0, "vy": 1022, "color": LIGHT_GRAY, "radius": 5, "name": "Moon"} # Moon orbiting Earth
    ],
    "Binary Star": [
        # Equal mass binary, stable orbit
        {"mass": 1.0 * SOLAR_MASS, "x": -0.5*AU/SPACE_SCALE, "y": 0, "vx": 0, "vy": 15000, "color": STAR_COLORS[0], "radius": 10, "name": "Star A"},
        {"mass": 1.0 * SOLAR_MASS, "x": 0.5*AU/SPACE_SCALE, "y": 0, "vx": 0, "vy": -15000, "color": STAR_COLORS[2], "radius": 10, "name": "Star B"}
    ],
    "Figure 8": [
        # Classic stable 3-body solution (requires precise initial conditions)
        # Using known initial conditions scaled appropriately
        # Note: Velocities are high, requires small time steps
        {"mass": SOLAR_MASS, "x": -0.97000436 * AU/SPACE_SCALE, "y": 0.24308753 * AU/SPACE_SCALE, "vx": 0.46620368 * 30000, "vy": 0.43236573 * 30000, "color": STAR_COLORS[0], "radius": 8, "name": "Body A"},
        {"mass": SOLAR_MASS, "x": 0, "y": 0, "vx": -0.93240737 * 30000, "vy": -0.86473146 * 30000, "color": STAR_COLORS[1], "radius": 8, "name": "Body B"},
        {"mass": SOLAR_MASS, "x": 0.97000436 * AU/SPACE_SCALE, "y": -0.24308753 * AU/SPACE_SCALE, "vx": 0.46620368 * 30000, "vy": 0.43236573 * 30000, "color": STAR_COLORS[2], "radius": 8, "name": "Body C"}
    ],
    "Inner Solar System": [
        # Simplified inner solar system (masses approximate, orbits circular)
        {"mass": SOLAR_MASS, "x": 0, "y": 0, "vx": 0, "vy": 0, "color": SUN_COLOR, "radius": 20, "name": "Sun", "fixed": True},
        {"mass": 3.301e23, "x": 0.387*AU/SPACE_SCALE, "y": 0, "vx": 0, "vy": 47870, "color": MERCURY_COLOR, "radius": 4, "name": "Mercury"},
        {"mass": 4.867e24, "x": 0.723*AU/SPACE_SCALE, "y": 0, "vx": 0, "vy": 35020, "color": VENUS_COLOR, "radius": 7, "name": "Venus"},
        {"mass": EARTH_MASS, "x": 1.0*AU/SPACE_SCALE, "y": 0, "vx": 0, "vy": 29780, "color": EARTH_COLOR, "radius": 8, "name": "Earth"},
        {"mass": 6.417e23, "x": 1.524*AU/SPACE_SCALE, "y": 0, "vx": 0, "vy": 24070, "color": MARS_COLOR, "radius": 6, "name": "Mars"}
    ],
     "Three-Body Chaos": [
        # Example of chaotic interaction
        {"mass": SOLAR_MASS, "x": 0, "y": 0, "vx": 0, "vy": 0, "color": SUN_COLOR, "radius": 15, "name": "Central Mass", "fixed":True},
        {"mass": 1.0 * JUPITER_MASS, "x": 1.5*AU/SPACE_SCALE, "y": 0, "vx": 0, "vy": 15000, "color": JUPITER_COLOR, "radius": 10, "name": "Planet 1"},
        {"mass": 0.5 * JUPITER_MASS, "x": -2.0*AU/SPACE_SCALE, "y": 0.5*AU/SPACE_SCALE, "vx": 5000, "vy": -10000, "color": GAS_COLOR, "radius": 8, "name": "Planet 2"}
    ]
}

# --- JIT Compiled Physics Functions (if Numba available) ---
@nb.njit
def calculate_acceleration_jit(target_pos_sim, target_mass_kg, other_positions_sim, other_masses_kg, g_const, softening_sq_m2, space_scale):
    """
    Calculates gravitational acceleration on a target body. Numba-optimized.
    Works with simulation units for position and SI units for others.

    Args:
        target_pos_sim: Position [x, y] of the target body (simulation units).
        target_mass_kg: Mass of the target body (kg).
        other_positions_sim: Array of positions [[x1,y1], [x2,y2]...] of other bodies (simulation units).
        other_masses_kg: Array of masses [m1, m2...] of other bodies (kg).
        g_const: Gravitational constant (G_REAL, m^3 kg^-1 s^-2).
        softening_sq_m2: Softening factor squared (meters^2).
        space_scale: Conversion factor from simulation units to meters.

    Returns:
        Acceleration vector [ax, ay] (m/s^2).
    """
    acc = np.zeros(2, dtype=np.float64)
    num_others = len(other_masses_kg)

    for i in range(num_others):
        # Check if the 'other' body is actually the target body by comparing positions
        # Note: Floating point comparison needs tolerance, but exact match check is faster if applicable.
        # If target body data is included in 'others', skip self-interaction.
        # Assuming target body is NOT included in 'others' array for simplicity here.
        # If it IS included, add: if np.all(target_pos_sim == other_positions_sim[i]): continue

        # Vector from target to other body (in simulation units)
        distance_vec_sim = other_positions_sim[i] - target_pos_sim
        # Squared distance (in simulation units squared)
        dist_sq_sim = distance_vec_sim[0]**2 + distance_vec_sim[1]**2

        if dist_sq_sim == 0:
            continue # Avoid division by zero if bodies are at the exact same simulation position

        # Convert simulation distance squared to meters squared
        dist_sq_meters = dist_sq_sim * (space_scale**2)

        # Calculate force magnitude: G * m_other / (r_m^2 + softening_m^2)
        # Note: We calculate acceleration directly (Force / target_mass), so target_mass cancels if included here.
        # Acceleration = G * m_other / (r_m^2 + softening_m^2)
        acc_mag = g_const * other_masses_kg[i] / (dist_sq_meters + softening_sq_m2) # m/s^2

        # Direction vector (unit vector from target to other) - use simulation units for direction
        dist_sim = np.sqrt(dist_sq_sim)
        direction_vec = distance_vec_sim / dist_sim # Normalized direction vector

        # Add acceleration component from this body
        acc += direction_vec * acc_mag

    return acc

@nb.njit
def apply_boundary_conditions_jit(pos_sim, vel_m_s, bounds_sim, elasticity):
    """
    Apply reflecting boundary conditions. Numba-optimized.

    Args:
        pos_sim: Position vector [x, y] (simulation units).
        vel_m_s: Velocity vector [vx, vy] (m/s).
        bounds_sim: [min_x, min_y, max_x, max_y] boundary limits (simulation units).
        elasticity: Coefficient of restitution (0-1).

    Returns:
        Updated position (sim units) and velocity (m/s) vectors.
    """
    min_x, min_y, max_x, max_y = bounds_sim
    pos_out = pos_sim.copy()
    vel_out = vel_m_s.copy()
    collided = False

    # Check X boundaries
    if pos_out[0] < min_x:
        pos_out[0] = min_x + (min_x - pos_out[0]) * elasticity # Reflect position slightly
        vel_out[0] *= -elasticity
        collided = True
    elif pos_out[0] > max_x:
        pos_out[0] = max_x - (pos_out[0] - max_x) * elasticity
        vel_out[0] *= -elasticity
        collided = True

    # Check Y boundaries
    if pos_out[1] < min_y:
        pos_out[1] = min_y + (min_y - pos_out[1]) * elasticity
        vel_out[1] *= -elasticity
        collided = True
    elif pos_out[1] > max_y:
        pos_out[1] = max_y - (pos_out[1] - max_y) * elasticity
        vel_out[1] *= -elasticity
        collided = True

    # If collided, slightly move position back inside boundary to prevent sticking
    if collided:
        pos_out[0] = max(min_x, min(pos_out[0], max_x))
        pos_out[1] = max(min_y, min(pos_out[1], max_y))


    return pos_out, vel_out

# --- Body Class ---
class Body:
    """Represents a celestial body with physical and visual properties."""
    ID_counter = 0  # Class variable for unique IDs

    def __init__(self, mass, x, y, vx, vy, color, radius, max_trail_length=DEFAULT_TRAIL_LENGTH,
                 fixed=False, name=None, show_trail=True):
        # Physical properties
        self.mass = float(mass)  # Mass (kg)
        # Position uses simulation units for easier drawing/scaling
        self.pos = np.array([float(x), float(y)], dtype=np.float64) # Simulation units
        self.vel = np.array([float(vx), float(vy)], dtype=np.float64) # Velocity (m/s)
        self.acc = np.zeros(2, dtype=np.float64)  # Acceleration (m/s^2) - calculated each step
        self.fixed = fixed  # If true, position and velocity do not change

        # Visual properties
        self.color = color
        self.radius_pixels = max(1, int(radius)) # Base visual radius in pixels
        self.show_trail = show_trail
        # Trail stores screen positions (pixels) for direct drawing
        self.trail = deque(maxlen=max_trail_length)
        self.visible = True # Can be used to hide without removing

        # Metadata
        self.id = Body.ID_counter
        Body.ID_counter += 1
        self.name = name if name else f"Body {self.id}"

        # Internal state for physics/drawing
        self.last_screen_pos = np.zeros(2) # Cache last screen position

    def update_physics_state(self, new_pos_sim, new_vel_m_s):
        """Updates the body's physical state if not fixed."""
        if not self.fixed:
            self.pos = new_pos_sim
            self.vel = new_vel_m_s

    def update_trail(self, zoom, pan_offset):
        """Adds the current screen position to the trail deque."""
        if not self.show_trail or not self.visible:
            if len(self.trail) > 0: self.trail.clear()
            return

        # Calculate current screen position (pixels)
        screen_pos = self.pos * zoom + pan_offset
        self.last_screen_pos = screen_pos # Cache for drawing
        self.trail.append(screen_pos.copy())

    def clear_trail(self):
        """Clears the trail deque."""
        self.trail.clear()

    def draw(self, screen, zoom, pan_offset, draw_labels):
        """Draws the body and its trail onto the screen."""
        # global DEBUG_DRAW_COUNT # Access debug counter

        if not self.visible:
            return

        # Use cached screen position if available (calculated in update_trail)
        screen_pos = self.last_screen_pos
        draw_pos = (int(screen_pos[0]), int(screen_pos[1]))

        # Basic culling: check if center is way off-screen
        margin = 100 # Pixel margin
        sim_width_pixels = WIDTH - UI_SIDEBAR_WIDTH
        sim_height_pixels = HEIGHT - UI_BOTTOM_HEIGHT
        if (draw_pos[0] < -margin or draw_pos[0] > sim_width_pixels + margin or
            draw_pos[1] < -margin or draw_pos[1] > sim_height_pixels + margin):
            # If center is off-screen, still need to draw trail if it enters screen
            pass # Don't return yet, trail might be visible

        # Draw trail first (behind body)
        if self.show_trail and len(self.trail) > 1:
            trail_points_pixels = list(self.trail)
            num_points = len(trail_points_pixels)
            # <<< Draw continuous line trail with fading alpha >>>
            max_segments = 100 # Max line segments to draw for performance
            step = max(1, num_points // max_segments)

            for i in range(0, num_points - step, step):
                start_idx = i
                end_idx = i + step
                # Ensure indices are within bounds
                if end_idx >= num_points: end_idx = num_points - 1
                if start_idx >= end_idx: continue

                start_pos = (int(trail_points_pixels[start_idx][0]), int(trail_points_pixels[start_idx][1]))
                end_pos = (int(trail_points_pixels[end_idx][0]), int(trail_points_pixels[end_idx][1]))

                # Calculate alpha based on position in trail (fade out towards tail)
                alpha = int(150 * (1.0 - (i / num_points))) # Fade out
                alpha = max(0, min(255, alpha)) # Clamp alpha

                if alpha > 10: # Don't draw fully transparent lines
                    try:
                        # Draw anti-aliased line with alpha
                        pygame.draw.aaline(screen, (*self.color, alpha), start_pos, end_pos)
                    except TypeError: # Fallback if alpha not supported by aaline version
                        pygame.draw.line(screen, self.color, start_pos, end_pos, 1)
                    except Exception as e: # Catch other potential drawing errors
                        # print(f"Trail drawing error: {e}") # Debug
                        pass


        # Draw the body itself (on top of trail)
        # Scale visual radius with zoom^power, ensure minimum size
        effective_zoom_scale = max(0.1, zoom**BODY_ZOOM_SCALING_POWER) # Prevent zero/negative scale
        # Increased minimum radius slightly
        draw_radius = max(3, int(self.radius_pixels * effective_zoom_scale))

        # <<< DEBUG PRINT for Zoom Scaling Removed >>>

        # Check if body circle itself is on screen before drawing
        if (draw_pos[0] + draw_radius < 0 or draw_pos[0] - draw_radius > sim_width_pixels or
            draw_pos[1] + draw_radius < 0 or draw_pos[1] - draw_radius > sim_height_pixels):
             return # Body circle is completely off-screen

        # Simple glow effect for larger bodies
        if draw_radius > 4:
            glow_radius = draw_radius + int(draw_radius * 0.4)
            glow_alpha = 80 # Semi-transparent glow
            glow_color = (*self.color, glow_alpha)
            # Draw using gfxdraw for alpha circle (requires surface)
            try:
                # Create a temporary surface for the glow circle
                max_r = glow_radius
                glow_surface = pygame.Surface((max_r*2, max_r*2), pygame.SRCALPHA)
                # Draw filled circle with alpha on the temp surface
                pygame.gfxdraw.filled_circle(glow_surface, max_r, max_r, max_r, glow_color)
                # Blit the glow surface centered on the body position
                screen.blit(glow_surface, (draw_pos[0] - max_r, draw_pos[1] - max_r))
            except Exception as e:
                # print(f"Glow drawing error: {e}") # Debug
                pass # Skip glow if error occurs


        # Draw the main body circle (filled + anti-aliased outline)
        try:
            pygame.gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], draw_radius, self.color)
            pygame.gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], draw_radius, self.color) # Outline
        except Exception as e:
             # print(f"Body drawing error: {e}") # Debug
             pygame.draw.circle(screen, self.color, draw_pos, draw_radius) # Fallback


        # Draw label if enabled and body is large enough
        if draw_labels and draw_radius > 5:
            try:
                font_size = max(12, min(18, int(10 * zoom**0.5))) # Scale font size slightly with zoom
                font = pygame.font.Font(None, font_size)
                label = font.render(self.name, True, WHITE)
                label_pos = (draw_pos[0] - label.get_width() // 2, draw_pos[1] + draw_radius + 2) # Below body
                screen.blit(label, label_pos)
            except Exception as e:
                # print(f"Label drawing error: {e}") # Debug
                pass # Skip label if error

    def get_screen_pos(self, zoom, pan_offset):
        """Calculates the body's center position on the screen (pixels)."""
        screen_pos = self.pos * zoom + pan_offset
        return (int(screen_pos[0]), int(screen_pos[1]))

    def handle_boundary_collision(self, bounds_sim, elasticity=0.8):
        """Applies boundary conditions using JIT function."""
        if self.fixed: return

        # Pass current state and bounds to JIT function
        new_pos, new_vel = apply_boundary_conditions_jit(
            self.pos, self.vel, bounds_sim, elasticity
        )
        # Update body state only if changed
        if not np.array_equal(new_pos, self.pos) or not np.array_equal(new_vel, self.vel):
             self.pos = new_pos
             self.vel = new_vel
             # Optional: Count collisions, change color, etc.


# --- Physics Calculation Functions ---

def calculate_system_energies(bodies, g_constant):
    """
    Calculates total kinetic, potential, and total energy of the system.

    Args:
        bodies: List of Body objects.
        g_constant: Gravitational constant (G_REAL * multiplier).

    Returns:
        Tuple (kinetic_energy, potential_energy, total_energy) in Joules.
    """
    kinetic = 0.0
    potential = 0.0

    # Kinetic Energy: Sum(0.5 * m * v^2)
    for body in bodies:
        if body.fixed: continue
        speed_sq = np.dot(body.vel, body.vel) # vel is in m/s
        kinetic += 0.5 * body.mass * speed_sq

    # Potential Energy: Sum(-G * m1 * m2 / r) for all pairs
    num_bodies = len(bodies)
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            body1, body2 = bodies[i], bodies[j]

            # Distance vector in simulation units
            distance_vec_sim = body2.pos - body1.pos
            dist_sq_sim = np.dot(distance_vec_sim, distance_vec_sim)

            if dist_sq_sim > 1e-18: # Avoid division by zero / extreme values if bodies overlap exactly
                dist_sim = np.sqrt(dist_sq_sim)
                dist_meters = dist_sim * SPACE_SCALE # Convert to meters

                # Use softened distance for potential? Usually PE uses raw distance.
                # Let's use raw distance for PE calculation.
                potential -= g_constant * body1.mass * body2.mass / dist_meters

    return kinetic, potential, kinetic + potential

def calculate_system_momentum(bodies):
    """Calculates the total linear momentum vector of the system (kg*m/s)."""
    total_momentum = np.zeros(2, dtype=np.float64)
    for body in bodies:
        if body.fixed: continue
        total_momentum += body.mass * body.vel # vel is in m/s
    return total_momentum

def calculate_center_of_mass(bodies):
    """
    Calculates the center of mass position and velocity of the system.

    Args:
        bodies: List of Body objects.

    Returns:
        Tuple (com_pos_sim, com_vel_m_s). Returns (None, None) if no mass.
        com_pos_sim is in simulation units.
        com_vel_m_s is in m/s.
    """
    total_mass = 0.0
    weighted_pos_sum = np.zeros(2, dtype=np.float64)
    weighted_vel_sum = np.zeros(2, dtype=np.float64)

    has_mass = False
    for body in bodies:
        if not body.fixed and body.mass > 0:
            has_mass = True
            total_mass += body.mass
            weighted_pos_sum += body.pos * body.mass # pos is sim units
            weighted_vel_sum += body.vel * body.mass # vel is m/s

    if not has_mass or total_mass == 0:
        # Handle case with no moving mass (e.g., only fixed bodies or empty)
        # Return geometric center of view? Or center of fixed bodies?
        fixed_bodies = [b for b in bodies if b.fixed]
        if fixed_bodies:
             # Calculate average position of fixed bodies if they exist
             if len(fixed_bodies) > 0:
                 com_pos_sim = sum(b.pos for b in fixed_bodies) / len(fixed_bodies)
             else: # Should not happen if has_mass is False and fixed_bodies is checked
                 com_pos_sim = np.array([0.0, 0.0]) # Default fallback
             com_vel_m_s = np.zeros(2) # Fixed bodies have zero velocity
             return com_pos_sim, com_vel_m_s
        else:
            # No bodies at all, return None
            return None, None


    com_pos_sim = weighted_pos_sum / total_mass
    com_vel_m_s = weighted_vel_sum / total_mass

    return com_pos_sim, com_vel_m_s


# --- RK4 Integrator ---

def calculate_accelerations_for_all(bodies, g_constant):
    """Calculates acceleration for all non-fixed bodies."""
    # Removed debug prints

    accelerations = np.zeros((len(bodies), 2), dtype=np.float64)
    if len(bodies) < 2: return accelerations # No forces if less than 2 bodies

    positions_sim = np.array([b.pos for b in bodies])
    masses_kg = np.array([b.mass for b in bodies])

    for i, body in enumerate(bodies):
        if body.fixed:
            continue # Acceleration remains zero

        # --- Non-JIT calculation loop ---
        acc_i = np.zeros(2, dtype=np.float64)
        for j, other_body in enumerate(bodies):
            if i == j: continue # Skip self

            dist_vec_sim = other_body.pos - body.pos
            dist_sq_sim = np.dot(dist_vec_sim, dist_vec_sim)
            if dist_sq_sim == 0: continue

            dist_sq_meters = dist_sq_sim * (SPACE_SCALE**2)
            # Use the provided g_constant (which might be scaled by multiplier)
            # Add small epsilon to denominator to prevent division by zero with zero softening
            acc_mag = g_constant * other_body.mass / (dist_sq_meters + SOFTENING_FACTOR_SQ + 1e-18)

            dist_sim = np.sqrt(dist_sq_sim)
            direction = dist_vec_sim / dist_sim

            acc_i += direction * acc_mag
        accelerations[i] = acc_i

    return accelerations


def perform_rk4_step(bodies, dt, g_constant):
    """
    Performs a single RK4 step for all non-fixed bodies.
    Returns proposed new positions (sim units) and velocities (m/s).
    Does NOT update the bodies themselves.

    Args:
        bodies: List of Body objects.
        dt: Time step (seconds).
        g_constant: Gravitational constant (potentially scaled).

    Returns:
        Tuple (new_positions_sim, new_velocities_m_s) as numpy arrays.
    """
    n = len(bodies)
    if n == 0: return np.array([]), np.array([])

    initial_pos_sim = np.array([b.pos for b in bodies])
    initial_vel_m_s = np.array([b.vel for b in bodies])
    fixed_mask = np.array([b.fixed for b in bodies])

    # --- RK4 Calculation ---
    # k1: Derivatives at the initial state
    k1_acc = calculate_accelerations_for_all(bodies, g_constant) # m/s^2
    k1_vel = initial_vel_m_s.copy() # m/s

    # k2: Derivatives at midpoint using k1
    mid_pos_k2 = initial_pos_sim + (k1_vel * (0.5 * dt)) / SPACE_SCALE # Sim units
    mid_vel_k2 = initial_vel_m_s + k1_acc * (0.5 * dt) # m/s
    # Create temporary body states for acceleration calculation
    temp_bodies_k2 = [{'pos': p, 'vel': v, 'mass': b.mass, 'fixed': b.fixed}
                      for p, v, b in zip(mid_pos_k2, mid_vel_k2, bodies)]
    k2_acc = calculate_accelerations_from_temp(temp_bodies_k2, g_constant) # m/s^2
    k2_vel = mid_vel_k2 # m/s

    # k3: Derivatives at midpoint using k2
    mid_pos_k3 = initial_pos_sim + (k2_vel * (0.5 * dt)) / SPACE_SCALE # Sim units
    mid_vel_k3 = initial_vel_m_s + k2_acc * (0.5 * dt) # m/s
    temp_bodies_k3 = [{'pos': p, 'vel': v, 'mass': b.mass, 'fixed': b.fixed}
                      for p, v, b in zip(mid_pos_k3, mid_vel_k3, bodies)]
    k3_acc = calculate_accelerations_from_temp(temp_bodies_k3, g_constant) # m/s^2
    k3_vel = mid_vel_k3 # m/s

    # k4: Derivatives at endpoint using k3
    end_pos_k4 = initial_pos_sim + (k3_vel * dt) / SPACE_SCALE # Sim units
    end_vel_k4 = initial_vel_m_s + k3_acc * dt # m/s
    temp_bodies_k4 = [{'pos': p, 'vel': v, 'mass': b.mass, 'fixed': b.fixed}
                      for p, v, b in zip(end_pos_k4, end_vel_k4, bodies)]
    k4_acc = calculate_accelerations_from_temp(temp_bodies_k4, g_constant) # m/s^2
    k4_vel = end_vel_k4 # m/s

    # Combine derivatives for final update
    final_pos_sim = initial_pos_sim.copy()
    final_vel_m_s = initial_vel_m_s.copy()

    # Calculate change in position (sim units) and velocity (m/s)
    # dPos/dt = Vel / SPACE_SCALE (sim units / s)
    # dVel/dt = Acc (m/s^2)
    pos_update = (dt / 6.0) * (k1_vel/SPACE_SCALE + 2*k2_vel/SPACE_SCALE + 2*k3_vel/SPACE_SCALE + k4_vel/SPACE_SCALE)
    vel_update = (dt / 6.0) * (k1_acc + 2*k2_acc + 2*k3_acc + k4_acc)

    # Apply updates only to non-fixed bodies
    final_pos_sim[~fixed_mask] += pos_update[~fixed_mask]
    final_vel_m_s[~fixed_mask] += vel_update[~fixed_mask]

    return final_pos_sim, final_vel_m_s

# Helper for RK4 using temporary structures (list of dicts)
def calculate_accelerations_from_temp(temp_bodies_list, g_constant):
    """Calculates accelerations based on a list of temporary body states (dicts)."""
    num_bodies = len(temp_bodies_list)
    accelerations = np.zeros((num_bodies, 2), dtype=np.float64)
    if num_bodies < 2: return accelerations

    # Extract data into arrays for potential JIT use later if refactored
    positions_sim = np.array([tb['pos'] for tb in temp_bodies_list])
    masses_kg = np.array([tb['mass'] for tb in temp_bodies_list])

    for i, current_tb in enumerate(temp_bodies_list):
        if current_tb['fixed']: continue

        acc_i = np.zeros(2, dtype=np.float64)
        for j, other_tb in enumerate(temp_bodies_list):
            if i == j: continue

            dist_vec_sim = other_tb['pos'] - current_tb['pos']
            dist_sq_sim = np.dot(dist_vec_sim, dist_vec_sim)
            if dist_sq_sim == 0: continue

            dist_sq_meters = dist_sq_sim * (SPACE_SCALE**2)
            # Add small epsilon to denominator
            acc_mag = g_constant * other_tb['mass'] / (dist_sq_meters + SOFTENING_FACTOR_SQ + 1e-18)

            dist_sim = np.sqrt(dist_sq_sim)
            direction = dist_vec_sim / dist_sim
            acc_i += direction * acc_mag

        accelerations[i] = acc_i

    return accelerations


def adaptive_rk4_step(bodies, current_dt, g_constant, error_tolerance, use_boundaries, bounds_sim):
    """
    Performs an adaptive RK4 step. Updates bodies in-place if step is accepted.

    Args:
        bodies: List of Body objects.
        current_dt: Current time step size (seconds).
        g_constant: Gravitational constant (potentially scaled).
        error_tolerance: Target relative error.
        use_boundaries: Boolean flag.
        bounds_sim: Boundary limits [min_x, min_y, max_x, max_y] (simulation units).

    Returns:
        Tuple (time_advanced, next_dt_suggestion).
        time_advanced is the actual time the simulation progressed (usually current_dt if accepted).
        next_dt_suggestion is the recommended step size for the next iteration.
    """
    if not bodies: return 0.0, current_dt

    # Clamp proposed dt by min/max limits
    dt = max(MIN_TIME_STEP, min(current_dt, MAX_TIME_STEP))

    # --- Error Estimation ---
    # 1. Perform one step with dt
    pos1, vel1 = perform_rk4_step(bodies, dt, g_constant)

    # 2. Perform two steps with dt/2
    half_dt = dt / 2.0
    # First half-step from initial state
    pos_half, vel_half = perform_rk4_step(bodies, half_dt, g_constant)
    # Create temporary state after first half-step
    temp_bodies_half = [{'pos': p, 'vel': v, 'mass': b.mass, 'fixed': b.fixed}
                        for p, v, b in zip(pos_half, vel_half, bodies)]
    # Second half-step from intermediate state (need a perform_rk4_step variant for temp bodies)
    # Let's reuse perform_rk4_step by creating temporary Body objects (less efficient but simpler)
    temp_body_objects = []
    for i, b in enumerate(bodies):
         tb = Body(b.mass, 0,0,0,0, b.color, b.radius_pixels, fixed=b.fixed) # Dummy visual data
         tb.pos = pos_half[i]
         tb.vel = vel_half[i]
         temp_body_objects.append(tb)
    pos2, vel2 = perform_rk4_step(temp_body_objects, half_dt, g_constant)


    # --- Calculate Error ---
    # Compare results: pos1 (full step) vs pos2 (two half steps)
    max_rel_error = 0.0
    initial_pos_sim = np.array([b.pos for b in bodies])
    initial_vel_m_s = np.array([b.vel for b in bodies])

    for i, body in enumerate(bodies):
        if body.fixed: continue

        # Error = Difference between the two estimates
        pos_error_sim = np.linalg.norm(pos2[i] - pos1[i])
        vel_error_m_s = np.linalg.norm(vel2[i] - vel1[i])

        # Scale factor for relative error (use final state + initial state + small constant)
        pos_scale = np.linalg.norm(pos2[i]) + np.linalg.norm(initial_pos_sim[i]) + 1e-9 * SPACE_SCALE # Sim units scale
        vel_scale = np.linalg.norm(vel2[i]) + np.linalg.norm(initial_vel_m_s[i]) + 1e-6 # m/s scale

        # Relative error estimate (error / scale)
        # Note: RK45 error estimate is typically scaled by dt, but this simpler comparison works
        rel_pos_error = pos_error_sim / pos_scale if pos_scale > 1e-15 else 0
        rel_vel_error = vel_error_m_s / vel_scale if vel_scale > 1e-15 else 0

        current_body_error = max(rel_pos_error, rel_vel_error)
        max_rel_error = max(max_rel_error, current_body_error)

    # --- Step Size Control ---
    safety_factor = 0.9
    # Optimal step size scaling factor based on error ratio (p=5 for RK4)
    if max_rel_error <= 1e-15: # Avoid division by zero / instability if error is tiny
        scale_factor = 2.0 # Increase step size significantly
    else:
        scale_factor = safety_factor * (error_tolerance / max_rel_error)**0.2

    dt_new = dt * scale_factor
    dt_new = max(MIN_TIME_STEP, min(dt_new, MAX_TIME_STEP)) # Clamp suggestion

    # --- Accept or Reject Step ---
    if max_rel_error <= error_tolerance or dt <= MIN_TIME_STEP:
        # Step accepted! Update bodies with the more accurate result (pos2, vel2).
        for i, body in enumerate(bodies):
            if not body.fixed:
                body.update_physics_state(pos2[i], vel2[i])
                # Apply boundary conditions AFTER updating state
                if use_boundaries and bounds_sim is not None:
                    body.handle_boundary_collision(bounds_sim)

        # Return time advanced (dt) and the suggestion for the next step (dt_new)
        return dt, dt_new
    else:
        # Step rejected. Error too large.
        # Do not update bodies. Return 0 time advanced and the reduced step size suggestion (dt_new).
        # The main loop should retry with dt_new.
        return 0.0, dt_new


# --- Collision Detection ---
def detect_and_handle_collisions(bodies, merge_on_collision=False):
    """
    Detects collisions based on physical radius and handles them (bounce or merge).
    Modifies bodies list if merging occurs. Returns list of indices removed.
    """
    # global DEBUG_COLLISION_COUNT # Access debug counter

    num_bodies = len(bodies)
    if num_bodies < 2: return []

    collided_pairs = set()
    indices_to_remove = []

    # Calculate physical radius for each body (e.g., based on mass/density)
    # Assume constant density: Radius ~ Mass^(1/3)
    # Scale relative to Earth radius/mass
    earth_radius_sim = EARTH_RADIUS_METERS / SPACE_SCALE
    physical_radii_sim = []
    for body in bodies:
         if body.mass <= 0: # Handle zero or negative mass
              radius_sim = 0.001 * earth_radius_sim # Assign a tiny radius
         else:
              mass_ratio = body.mass / EARTH_MASS
              radius_sim = earth_radius_sim * (mass_ratio**(1/3))
         physical_radii_sim.append(max(radius_sim, 0.001 * earth_radius_sim)) # Ensure a minimum radius


    for i in range(num_bodies):
        if i in indices_to_remove: continue # Skip if already marked for removal
        for j in range(i + 1, num_bodies):
            if j in indices_to_remove: continue # Skip if already marked for removal

            body1, body2 = bodies[i], bodies[j]
            radius1_sim, radius2_sim = physical_radii_sim[i], physical_radii_sim[j]

            # Distance vector and squared distance (simulation units)
            distance_vec_sim = body2.pos - body1.pos
            dist_sq_sim = np.dot(distance_vec_sim, distance_vec_sim)

            # Collision threshold squared (sum of physical radii * factor squared)
            collision_threshold = (radius1_sim + radius2_sim) * COLLISION_DISTANCE_FACTOR
            collision_threshold_sq = collision_threshold**2

            # <<< DEBUG PRINT for Collision Check Removed >>>

            # Check for collision
            if dist_sq_sim < collision_threshold_sq and dist_sq_sim > 1e-18: # Check overlap and avoid exact same position
                pair = tuple(sorted((i, j)))
                if pair in collided_pairs: continue # Already handled this pair

                dist_sim = np.sqrt(dist_sq_sim)
                # Collision detected! Handle it.
                if merge_on_collision:
                    # --- Merge Logic ---
                    if body1.fixed or body2.fixed: continue # Don't merge fixed bodies

                    # Determine survivor (larger mass) and removed body
                    if body1.mass >= body2.mass:
                        survivor, removed = body1, body2
                        survivor_idx, removed_idx = i, j
                    else:
                        survivor, removed = body2, body1
                        survivor_idx, removed_idx = j, i

                    # Conserve momentum
                    total_mass = survivor.mass + removed.mass
                    # Avoid division by zero if total mass is zero (though unlikely here)
                    if total_mass == 0: continue
                    new_vel = (survivor.mass * survivor.vel + removed.mass * removed.vel) / total_mass

                    # New position is center of mass
                    new_pos = (survivor.mass * survivor.pos + removed.mass * removed.pos) / total_mass

                    # Update survivor properties
                    survivor.mass = total_mass
                    survivor.pos = new_pos
                    survivor.vel = new_vel
                    # Update visual radius (approx volume conservation)
                    survivor.radius_pixels = (body1.radius_pixels**3 + body2.radius_pixels**3)**(1/3)
                    survivor.name += f"+{removed.name}" # Append name
                    survivor.clear_trail()

                    # Mark the removed body's index for deletion
                    if removed_idx not in indices_to_remove:
                         indices_to_remove.append(removed_idx)
                    collided_pairs.add(pair)
                    print(f"Collision: Merged {removed.name} into {survivor.name}")

                else:
                    # --- Bounce Logic ---
                    if body1.fixed and body2.fixed: continue # Cannot bounce

                    # Normal vector (from body1 to body2)
                    normal_sim = distance_vec_sim / dist_sim # Unit vector

                    # Relative velocity (v2 - v1) in m/s
                    rel_vel_m_s = body2.vel - body1.vel

                    # Velocity component along the normal (m/s)
                    vel_along_normal = np.dot(rel_vel_m_s, normal_sim)

                    # If separating already, do nothing
                    if vel_along_normal > 0: continue

                    # Coefficient of restitution (elasticity)
                    cor = 0.7 # Slightly inelastic bounce

                    # Calculate impulse scalar (j) - handle fixed bodies and zero mass
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
                        # Both bodies move
                        if body1.mass > 0 and body2.mass > 0:
                            inv_mass_sum = (1 / body1.mass) + (1 / body2.mass)
                            impulse = -(1 + cor) * vel_along_normal / inv_mass_sum
                            body1.vel -= (impulse / body1.mass) * normal_sim
                            body2.vel += (impulse / body2.mass) * normal_sim
                        elif body1.mass > 0: # Only body 2 has zero mass (treat as fixed?)
                             impulse = -(1 + cor) * vel_along_normal / (1 / body1.mass)
                             body1.vel -= (impulse / body1.mass) * normal_sim
                        elif body2.mass > 0: # Only body 1 has zero mass
                             impulse = -(1 + cor) * vel_along_normal / (1 / body2.mass)
                             body2.vel += (impulse / body2.mass) * normal_sim
                        # If both zero mass, impulse is zero, no change

                    # --- Position Correction (to prevent sticking) ---
                    overlap_sim = collision_threshold - dist_sim # Use threshold without factor here? No, use factored one.
                    if overlap_sim > 0:
                        correction_factor = 0.6 # How much overlap to correct instantly
                        correction_vec = normal_sim * overlap_sim * correction_factor

                        if body1.fixed:
                            body2.pos += correction_vec * 2 # Move only body2
                        elif body2.fixed:
                            body1.pos -= correction_vec * 2 # Move only body1
                        else:
                            # Move proportional to inverse mass (handle zero mass)
                            m1 = body1.mass
                            m2 = body2.mass
                            total_mass = m1 + m2
                            if total_mass > 0: # Avoid division by zero
                                body1.pos -= correction_vec * (m2 / total_mass) * 2
                                body2.pos += correction_vec * (m1 / total_mass) * 2
                            elif m1 > 0: # Only body 1 has mass, move only body 2
                                 body2.pos += correction_vec * 2
                            elif m2 > 0: # Only body 2 has mass, move only body 1
                                 body1.pos -= correction_vec * 2
                            # If both zero mass, no position correction needed? Or arbitrary split?


                    collided_pairs.add(pair)

    # Return list of indices that were merged and should be removed from main list
    return sorted(indices_to_remove, reverse=True) # Sort reverse for safe removal


# --- Visualization Functions ---

def render_gravitational_field(screen, bodies, g_constant, zoom, pan_offset):
    """Renders gravitational potential as a heatmap (if matplotlib available)."""
    # Access SHOW_GRAV_FIELD defined at module level
    global SHOW_GRAV_FIELD # Declare intent to potentially modify global

    if not MATPLOTLIB_AVAILABLE or not bodies: return

    resolution = FIELD_RESOLUTION
    sim_width_pixels = WIDTH - UI_SIDEBAR_WIDTH
    sim_height_pixels = HEIGHT - UI_BOTTOM_HEIGHT
    field_potential = np.zeros((resolution, resolution))

    # Calculate potential at grid cell centers
    for i in range(resolution): # x index
        for j in range(resolution): # y index
            # Screen position of cell center
            screen_pos_x = (i + 0.5) * (sim_width_pixels / resolution)
            screen_pos_y = (j + 0.5) * (sim_height_pixels / resolution)
            screen_pos = np.array([screen_pos_x, screen_pos_y])

            # Convert screen position to simulation world position
            world_pos_sim = (screen_pos - pan_offset) / (zoom + 1e-18) # Avoid zoom=0

            # Calculate potential sum V = Sum(-G*m/r)
            potential_at_point = 0.0
            for body in bodies:
                dist_vec_sim = body.pos - world_pos_sim
                dist_sq_sim = np.dot(dist_vec_sim, dist_vec_sim)
                if dist_sq_sim > 1e-18: # Avoid issues at exact body location
                    dist_meters = np.sqrt(dist_sq_sim) * SPACE_SCALE
                    # Avoid division by zero if dist_meters is somehow zero
                    if dist_meters > 0:
                         potential_at_point -= g_constant * body.mass / dist_meters # J/kg

            field_potential[j, i] = potential_at_point # Store potential

    # Normalize potential for color mapping (handle log scale or clipping)
    # Using simple linear normalization for now
    min_pot = np.min(field_potential)
    max_pot = np.max(field_potential)

    if max_pot > min_pot:
         # Normalize, potentially clip outliers using percentiles?
         # Example: Clip to 1st and 99th percentile for better contrast
         p_low, p_high = np.percentile(field_potential, [1, 99])
         # Avoid p_high == p_low if field is almost flat
         if p_high <= p_low: p_high = p_low + 1e-9
         clipped_field = np.clip(field_potential, p_low, p_high)
         # Avoid division by zero if p_high == p_low after clipping
         range_pot = p_high - p_low
         if range_pot == 0: range_pot = 1.0 # Prevent division by zero
         field_normalized = (clipped_field - p_low) / range_pot


         # Alternative: Log scale normalization (careful with signs)
         # log_field = np.log(-field_potential + 1 - min_pot) # Shift to positive before log
         # field_normalized = (log_field - np.min(log_field)) / (np.max(log_field) - np.min(log_field))

    else:
        field_normalized = np.ones_like(field_potential) * 0.5 # Flat field


    # --- Create Pygame surface from normalized data ---
    try:
        cmap = cm.get_cmap('viridis') # Or 'plasma', 'inferno', 'magma'
        # Get RGBA values (0-1 range)
        rgba_values = cmap(field_normalized.flatten())
        # Convert to Pygame colors (0-255)
        pixels = np.array(rgba_values[:, :3] * 255, dtype=np.uint8) # RGB
        # Create surface and fill pixels (slow method)
        heatmap_surface = pygame.Surface((resolution, resolution))
        pixel_index = 0
        for y in range(resolution):
            for x in range(resolution):
                heatmap_surface.set_at((x, y), tuple(pixels[pixel_index])) # Use tuple for color
                pixel_index += 1

        # Scale up heatmap to screen size and set alpha
        scaled_heatmap = pygame.transform.smoothscale(heatmap_surface, (sim_width_pixels, sim_height_pixels))
        scaled_heatmap.set_alpha(100) # Adjust transparency

        # Blit onto the main screen
        screen.blit(scaled_heatmap, (0, 0))

    except Exception as e:
        print(f"Error rendering gravitational field: {e}")
        # Disable field rendering if error occurs repeatedly?
        SHOW_GRAV_FIELD = False # Disable on error


def get_world_bounds_sim(zoom, pan_offset):
    """Calculates the simulation-world bounds of the visible viewport."""
    sim_width_pixels = WIDTH - UI_SIDEBAR_WIDTH
    sim_height_pixels = HEIGHT - UI_BOTTOM_HEIGHT

    # Top-left screen corner in world coordinates
    min_screen = np.array([0, 0])
    min_world_sim = (min_screen - pan_offset) / (zoom + 1e-18) # Avoid zoom=0

    # Bottom-right screen corner in world coordinates
    max_screen = np.array([sim_width_pixels, sim_height_pixels])
    max_world_sim = (max_screen - pan_offset) / (zoom + 1e-18) # Avoid zoom=0

    return (min_world_sim[0], min_world_sim[1], max_world_sim[0], max_world_sim[1])


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
    boundaries_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((10, y_pos), (UI_SIDEBAR_WIDTH - 20, 30)), text="Boundaries: ON", manager=ui_manager, container=control_panel)
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

<b>UI Panel:</b><br>
 - Use sliders and buttons to control simulation parameters, visualization, add new bodies, and edit selected body mass.<br><br>

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
        nonlocal gravity_multiplier
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

            bodies.append(Body(mass, x_sim, y_sim, vx_m_s, vy_m_s, color, radius_px,
                               fixed=fixed, name=name, show_trail=SHOW_TRAILS))

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
            indices_to_remove = detect_and_handle_collisions(bodies, merge_on_collision=False) # Bounce default
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

