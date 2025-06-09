import argparse
from datetime import datetime
import pygame
import pygame_gui
import numpy as np
from pathlib import Path

from threebody import constants as C
from threebody.rendering import Body, render_gravitational_field
from threebody.presets import PRESETS
from threebody.physics_utils import (
    step_simulation,
    detect_and_handle_collisions,
    adaptive_rk4_step,
)
from threebody.analysis import EnergyMonitor
# Ensure the import paths for your new modules are correct
from threebody.state_manager import save_state, load_state
from threebody.ui_manager import ControlPanel, UIManager
from threebody.camera import Camera
from threebody.nasa import load_ephemeris, create_body


def _create_bodies(preset_name: str, ephem=None, epoch=None):
    """Create bodies either from a preset or NASA ephemeris."""
    if ephem is not None and epoch is not None:
        sun = create_body(ephem, 10, epoch, C.SOLAR_MASS, name="Sun")
        earth = create_body(ephem, 399, epoch, C.EARTH_MASS, name="Earth")
        return [sun, earth]

    bodies = []
    for cfg in PRESETS.get(preset_name, []):
        bodies.append(
            Body(
                cfg["mass"],
                [cfg.get("x", 0.0), cfg.get("y", 0.0)],
                [cfg.get("vx", 0.0), cfg.get("vy", 0.0)],
                cfg.get("color", C.WHITE),
                cfg.get("radius", C.DEFAULT_NEXT_BODY_RADIUS_PIXELS),
                fixed=cfg.get("fixed", False),
                name=cfg.get("name"),
            )
        )
    return bodies


def main(argv=None):
    parser = argparse.ArgumentParser(description="Three Body Simulation")
    parser.add_argument("--preset", default="Sun & Earth", help="Preset system")
    parser.add_argument(
        "--integrator",
        choices=["RK4", "Symplectic", "Symplectic4", "ForestRuth"],
        default="Symplectic",
    )
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive RK4")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument(
        "--gr",
        action="store_true",
        help="Use general relativity correction",
    )
    parser.add_argument("--show-field", action="store_true", help="Visualize gravitational field")
    parser.add_argument("--trail-length", type=int, help="Override body trail length")
    parser.add_argument("--merge", action="store_true", help="Merge bodies on collision")
    parser.add_argument("--nasa-kernel", help="Load Sun/Earth from SPK kernel")
    parser.add_argument("--nasa-date", help="Epoch YYYY-MM-DD for SPK data")
    args = parser.parse_args([] if argv is None else argv)

    pygame.init()
    screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
    pygame.display.set_caption("Three Body Simulation")

    # --- MERGED INITIALIZATION BLOCK ---
    # This combines the new objects from both branches, as all are needed.
    theme_path = Path(__file__).with_name("theme.json")
    manager = pygame_gui.UIManager((C.WIDTH, C.HEIGHT), theme_path) # From codex branch (for GUI)
    control = ControlPanel(manager, args.preset)                   # From codex branch (for UI widgets)
    ui = UIManager(
        manager,
        integrator=args.integrator,
        adaptive=args.adaptive,
        use_gr=args.use_gr,
        show_field=args.show_field,
    )
    camera = Camera()                                              # From main branch (for camera control)
    # --- MERGE CONFLICT FOR INITIALIZATION IS RESOLVED ---

    # These state variables should ideally be managed by the new classes, but are kept for now
    paused = False
    single_step = False
    
    ephem = None
    epoch = None
    if args.nasa_kernel:
        ephem = load_ephemeris(args.nasa_kernel)
        epoch = datetime.strptime(args.nasa_date or "2024-01-01", "%Y-%m-%d")

    bodies = _create_bodies(args.preset, ephem=ephem, epoch=epoch)

    if args.trail_length is not None:
        for b in bodies:
            if hasattr(b, "set_trail_length"):
                b.set_trail_length(args.trail_length)

    energy_monitor = EnergyMonitor()
    energy_monitor.set_initial_energy(bodies, C.G_REAL)
    clock = pygame.time.Clock()

    selected_body = None
    focus_body = None

    running = True
    while running:
        print(f"Bodies count: {len(bodies)}")
        print(f"Camera Zoom: {camera.zoom:.4f}, Camera Pan: {camera.pan_offset}")
        if bodies:
            print(f"Sun Position: {bodies[0].pos}, Earth Position: {bodies[1].pos}")
        time_delta = clock.tick(60) / 1000.0

        # --- MERGED EVENT HANDLING LOOP ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle keyboard and mouse clicks first
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    control.play_button.set_text("Pause" if not paused else "Play")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Only check for body selection if the mouse is not on a UI element
                if not manager.get_focus_set():
                    for b in bodies:
                        bx, by = b.get_screen_pos(camera.zoom, camera.pan_offset)
                        r = max(1, int(b.radius_pixels * (camera.zoom ** C.BODY_ZOOM_SCALING_POWER)))
                        if (event.pos[0]-bx)**2 + (event.pos[1]-by)**2 <= r*r:
                            selected_body = b
                            focus_body = b
                            break

            # Now, handle the specific pygame_gui events (from the codex branch)
            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == control.preset_menu:
                    bodies = _create_bodies(event.text, ephem=ephem, epoch=epoch)
                    for b in bodies:
                        if hasattr(b, "set_trail_length"):
                            b.set_trail_length(int(control.trail_slider.get_current_value()))
                    energy_monitor.set_initial_energy(bodies, C.G_REAL)
            elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == control.speed_slider:
                    C.TIME_STEP_BASE = event.value
                    control.update_speed_label(event.value)
                elif event.ui_element == control.trail_slider:
                    for b in bodies:
                        if hasattr(b, "set_trail_length"):
                            b.set_trail_length(int(event.value))
                    control.update_trail_label(event.value)
            elif event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == control.play_button:
                    paused = not paused
                    control.play_button.set_text("Pause" if not paused else "Play")
                elif event.ui_element == control.reset_button:
                    bodies = _create_bodies(control.preset_menu.selected_option, ephem=ephem, epoch=epoch)
                    energy_monitor.set_initial_energy(bodies, C.G_REAL)
                elif event.ui_element == control.step_button:
                    single_step = True
                elif event.ui_element == control.save_button:
                    save_state("simulation_save.json", bodies)
                elif event.ui_element == control.load_button:
                    bodies = load_state("simulation_save.json")
                    energy_monitor.set_initial_energy(bodies, C.G_REAL)
            
            # Finally, let the GUI manager process the event
            manager.process_events(event)
        # --- END OF EVENT HANDLING LOOP ---
        
        # --- MERGED UPDATE AND PHYSICS LOGIC ---
        manager.update(time_delta)
        ui.update(time_delta) 
        
        # Get simulation parameters from the state manager (from main branch)
        integrator = ui.integrator
        adaptive = ui.adaptive
        use_gr = ui.use_gr
        show_field = ui.show_field

        # Run physics simulation step (from main branch)
        if not paused or single_step:
            if adaptive and integrator == "RK4":
                dt, _ = adaptive_rk4_step(bodies, C.TIME_STEP_BASE, C.G_REAL, C.ERROR_TOLERANCE, False, None, use_gpu=args.use_gpu)
                if dt == 0.0: continue
            else:
                step_simulation(bodies, C.TIME_STEP_BASE, C.G_REAL, integrator_type=integrator, use_gr=use_gr, use_gpu=args.use_gpu)
            
            detect_and_handle_collisions(bodies, merge_on_collision=args.merge)
            energy_monitor.update(bodies, C.G_REAL)
            if single_step: single_step = False
        # --- END OF PHYSICS LOGIC ---

        # --- RENDERING ---
        camera.update_focus(focus_body, bodies)
        screen.fill(C.BLACK)

        if show_field:
            render_gravitational_field(screen, bodies, C.G_REAL, camera.zoom, camera.pan_offset)
        
        for b in bodies:
            if hasattr(b, "update_trail"): b.update_trail(camera.zoom, camera.pan_offset)
            if hasattr(b, "draw"): b.draw(screen, camera.zoom, camera.pan_offset, draw_labels=False)

        energy_monitor.draw(screen)

        # Update and draw the UI
        control.update_body_info(selected_body, bodies[0] if bodies else None)
        manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()