import pygame
from threebody import constants as C
from threebody.rendering import Body, render_gravitational_field
from threebody.presets import PRESETS
from threebody.physics_utils import step_simulation


def _create_bodies(preset_name: str):
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


def main():
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

    bodies = _create_bodies("Sun & Earth")
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
            step_simulation(bodies, C.TIME_STEP_BASE, C.G_REAL, integrator_type="RK4")
        screen.fill(C.BLACK)
        for b in bodies:
            if hasattr(b, "update_trail"):
                b.update_trail(zoom, pan_offset)
            if hasattr(b, "draw"):
                b.draw(screen, zoom, pan_offset, draw_labels=False)
        if show_help:
            for i, line in enumerate(help_lines):
                text = font.render(line, True, C.WHITE)
                screen.blit(text, (10, 10 + i * 20))
        if paused:
            pause_text = font.render("PAUSED", True, C.WHITE)
            screen.blit(pause_text, (10, C.HEIGHT - 30))
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, C.WHITE)
        screen.blit(fps_text, (C.WIDTH - 100, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
