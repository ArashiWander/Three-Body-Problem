import argparse
from datetime import datetime
import pygame
from threebody import constants as C
from threebody.rendering import Body, render_gravitational_field
from threebody.presets import PRESETS
from threebody.physics_utils import (
    step_simulation,
    detect_and_handle_collisions,
    get_world_bounds_sim,
    adaptive_rk4_step,
)
from threebody.analysis import EnergyMonitor
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
    parser.add_argument("--integrator", choices=["RK4", "Symplectic"], default="Symplectic")
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
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_h:
                    show_help = not show_help
        if not paused:
            if args.adaptive and args.integrator == "RK4":
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
                    integrator_type=args.integrator,
                    use_gr=args.gr,
                    use_gpu=args.use_gpu,
                )
            detect_and_handle_collisions(bodies, merge_on_collision=args.merge)
            energy_monitor.update(bodies, C.G_REAL)
        screen.fill(C.BLACK)
        for b in bodies:
            if hasattr(b, "update_trail"):
                b.update_trail(zoom, pan_offset)
            if hasattr(b, "draw"):
                b.draw(screen, zoom, pan_offset, draw_labels=False)
        if args.show_field:
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

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
