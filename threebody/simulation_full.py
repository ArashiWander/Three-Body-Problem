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
    get_world_bounds_sim,
    calculate_center_of_mass,
    adaptive_rk4_step,
)
from threebody.analysis import EnergyMonitor, calculate_orbital_elements
from threebody.state_io import save_state, load_state
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
    parser.add_argument("--gr", action="store_true", help="Use general relativity correction")
    parser.add_argument("--show-field", action="store_true", help="Visualize gravitational field")
    parser.add_argument("--trail-length", type=int, help="Override body trail length")
    parser.add_argument("--merge", action="store_true", help="Merge bodies on collision")
    parser.add_argument("--nasa-kernel", help="Load Sun/Earth from SPK kernel")
    parser.add_argument("--nasa-date", help="Epoch YYYY-MM-DD for SPK data")
    args = parser.parse_args([] if argv is None else argv)

    pygame.init()
    screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
    pygame.display.set_caption("Three Body Simulation")
    font = pygame.font.Font(None, 20)
    theme_path = Path(__file__).with_name("theme.json")
    manager = pygame_gui.UIManager((C.WIDTH, C.HEIGHT), theme_path)

    panel = pygame_gui.elements.UIPanel(
        pygame.Rect(C.WIDTH - C.UI_SIDEBAR_WIDTH, 0, C.UI_SIDEBAR_WIDTH, C.HEIGHT - C.UI_BOTTOM_HEIGHT),
        manager=manager,
        object_id="#control_panel",
    )

    pygame_gui.elements.UILabel(
        pygame.Rect(0, 0, panel.rect.width, 30),
        text="Controls",
        manager=manager,
        container=panel,
        object_id="#title_label",
    )

    integrator_menu = pygame_gui.elements.UIDropDownMenu(
        ["Symplectic", "RK4"],
        args.integrator,
        pygame.Rect(10, 40, panel.rect.width - 20, 25),
        manager=manager,
        container=panel,
    )
    adaptive_box = pygame_gui.elements.UICheckBox(
        pygame.Rect(10, 70, panel.rect.width - 20, 20),
        "Adaptive", manager, container=panel, initial_state=args.adaptive
    )
    gr_box = pygame_gui.elements.UICheckBox(
        pygame.Rect(10, 95, panel.rect.width - 20, 20),
        "GR Correction", manager, container=panel, initial_state=args.gr
    )
    field_box = pygame_gui.elements.UICheckBox(
        pygame.Rect(10, 120, panel.rect.width - 20, 20),
        "Show Field", manager, container=panel, initial_state=args.show_field
    )

    ts_label = pygame_gui.elements.UILabel(
        pygame.Rect(10, 145, panel.rect.width - 20, 20),
        f"dt: {C.TIME_STEP_BASE:.0f}", manager, container=panel
    )
    ts_slider = pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect(10, 165, panel.rect.width - 20, 20),
        start_value=C.TIME_STEP_BASE,
        value_range=(10, 3600),
        manager=manager,
        container=panel,
    )

    soft_label = pygame_gui.elements.UILabel(
        pygame.Rect(10, 190, panel.rect.width - 20, 20),
        f"soft: {C.SOFTENING_LENGTH:.2e}", manager, container=panel
    )
    soft_slider = pygame_gui.elements.UIHorizontalSlider(
        pygame.Rect(10, 210, panel.rect.width - 20, 20),
        start_value=C.SOFTENING_LENGTH,
        value_range=(0.1, 10.0),
        manager=manager,
        container=panel,
    )

    save_button = pygame_gui.elements.UIButton(
        pygame.Rect(10, 240, 60, 25), "Save", manager, container=panel
    )
    load_button = pygame_gui.elements.UIButton(
        pygame.Rect(80, 240, 60, 25), "Load", manager, container=panel
    )
    help_lines = [
        "SPACE: pause/resume",
        "H: toggle help",
    ]
    show_help = False
    paused = False

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
    zoom = C.ZOOM_BASE
    pan_offset = C.INITIAL_PAN_OFFSET
    clock = pygame.time.Clock()

    sim_width = C.WIDTH - C.UI_SIDEBAR_WIDTH
    sim_height = C.HEIGHT - C.UI_BOTTOM_HEIGHT
    screen_center = np.array([sim_width / 2, sim_height / 2], dtype=float)

    integrator = args.integrator
    adaptive = args.adaptive
    use_gr = args.gr
    show_field = args.show_field
    selected_body = None
    focus_body = None

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_c:
                    focus_body = 'COM'
                elif event.key == pygame.K_ESCAPE:
                    focus_body = None
                    selected_body = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for b in bodies:
                    bx, by = b.get_screen_pos(zoom, pan_offset)
                    r = max(1, int(b.radius_pixels * (zoom ** C.BODY_ZOOM_SCALING_POWER)))
                    if (event.pos[0]-bx)**2 + (event.pos[1]-by)**2 <= r*r:
                        selected_body = b
                        focus_body = b
                        break

            manager.process_events(event)

            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED and event.ui_element == integrator_menu:
                integrator = event.text
            if event.type in (pygame_gui.UI_CHECK_BOX_CHECKED, pygame_gui.UI_CHECK_BOX_UNCHECKED):
                state = event.ui_element.checked
                if event.ui_element == adaptive_box:
                    adaptive = state
                elif event.ui_element == gr_box:
                    use_gr = state
                elif event.ui_element == field_box:
                    show_field = state
            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == ts_slider:
                    C.TIME_STEP_BASE = event.value
                    ts_label.set_text(f"dt: {event.value:.0f}")
                elif event.ui_element == soft_slider:
                    C.SOFTENING_LENGTH = float(event.value)
                    C.SOFTENING_FACTOR_SQ = C.SOFTENING_LENGTH**2
                    soft_label.set_text(f"soft: {C.SOFTENING_LENGTH:.2e}")
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == save_button:
                    save_state("simulation_save.json", bodies)
                elif event.ui_element == load_button:
                    bodies = load_state("simulation_save.json")
                    energy_monitor.set_initial_energy(bodies, C.G_REAL)

        manager.update(time_delta)
        if not paused:
            if adaptive and integrator == "RK4":
                dt, _ = adaptive_rk4_step(
                    bodies,
                    C.TIME_STEP_BASE,
                    C.G_REAL,
                    C.ERROR_TOLERANCE,
                    False,
                    None,
                    use_gpu=args.use_gpu,
                )
                if dt == 0.0:
                    continue
            else:
                step_simulation(
                    bodies,
                    C.TIME_STEP_BASE,
                    C.G_REAL,
                    integrator_type=integrator,
                    use_gr=use_gr,
                    use_gpu=args.use_gpu,
                )
            detect_and_handle_collisions(bodies, merge_on_collision=args.merge)
            energy_monitor.update(bodies, C.G_REAL)

        if focus_body is not None:
            if focus_body == 'COM':
                com_pos, _ = calculate_center_of_mass(bodies)
                if com_pos is not None:
                    target = screen_center - com_pos[:2] * zoom
                    pan_offset += (target - pan_offset) * C.CAMERA_SMOOTHING
            else:
                target = screen_center - focus_body.pos[:2] * zoom
                pan_offset += (target - pan_offset) * C.CAMERA_SMOOTHING
        screen.fill(C.BLACK)
        for b in bodies:
            if hasattr(b, "update_trail"):
                b.update_trail(zoom, pan_offset)
            if hasattr(b, "draw"):
                b.draw(screen, zoom, pan_offset, draw_labels=False)
        if show_field:
            render_gravitational_field(screen, bodies, C.G_REAL, zoom, pan_offset)
        if show_help:
            for i, line in enumerate(help_lines):
                text = font.render(line, True, C.WHITE)
                screen.blit(text, (10, 10 + i * 20))
        if paused:
            pause_text = font.render("PAUSED", True, C.WHITE)
            screen.blit(pause_text, (10, C.HEIGHT - 30))
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, C.WHITE)
        screen.blit(fps_text, (C.WIDTH - 100, 10))
        energy_monitor.draw(screen)

        if selected_body is not None:
            elems = calculate_orbital_elements(selected_body, bodies[0]) if len(bodies) > 0 else {}
            info_lines = [
                f"{selected_body.name}",
                f"m={selected_body.mass:.2e} kg",
                f"pos={selected_body.pos}",
                f"vel={selected_body.vel}",
                f"a={elems.get('semi_major_axis',0):.2e} m e={elems.get('eccentricity',0):.3f}",
            ]
            for i, line in enumerate(info_lines):
                t = font.render(line, True, C.WHITE)
                screen.blit(t, (10, C.HEIGHT - 150 + i * 20))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
