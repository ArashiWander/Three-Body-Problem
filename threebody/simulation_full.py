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

    bodies = _create_bodies("Sun & Earth")
    zoom = C.ZOOM_BASE
    pan_offset = C.INITIAL_PAN_OFFSET
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        step_simulation(bodies, C.TIME_STEP_BASE, C.G_REAL, integrator_type="RK4")
        screen.fill(C.BLACK)
        for b in bodies:
            if hasattr(b, "update_trail"):
                b.update_trail(zoom, pan_offset)
            if hasattr(b, "draw"):
                b.draw(screen, zoom, pan_offset, draw_labels=False)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
