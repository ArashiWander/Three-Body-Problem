import pygame
import pygame_gui
from pathlib import Path
from . import constants as C
from .presets import PRESETS
from .analysis import calculate_orbital_elements


class ControlPanel:
    """Helper to build and manage the on-screen control panel."""

    def __init__(self, manager: pygame_gui.UIManager, default_preset: str):
        width = C.UI_SIDEBAR_WIDTH
        height = C.HEIGHT - C.UI_BOTTOM_HEIGHT
        self.manager = manager
        self.panel = pygame_gui.elements.UIPanel(
            pygame.Rect(C.WIDTH - width, 0, width, height),
            manager=manager,
            object_id="#control_panel",
        )
        y = 0
        pygame_gui.elements.UILabel(
            pygame.Rect(0, y, width, 30),
            text="Controls",
            manager=manager,
            container=self.panel,
            object_id="#title_label",
        )
        y += 40
        self.preset_menu = pygame_gui.elements.UIDropDownMenu(
            list(PRESETS.keys()),
            default_preset,
            pygame.Rect(10, y, width - 20, 25),
            manager=manager,
            container=self.panel,
        )
        y += 35
        self.play_button = pygame_gui.elements.UIButton(
            pygame.Rect(10, y, 80, 25),
            "Play",
            manager,
            container=self.panel,
        )
        self.reset_button = pygame_gui.elements.UIButton(
            pygame.Rect(100, y, 80, 25),
            "Reset",
            manager,
            container=self.panel,
        )
        self.step_button = pygame_gui.elements.UIButton(
            pygame.Rect(190, y, 80, 25),
            "Step",
            manager,
            container=self.panel,
        )
        y += 35
        self.speed_label = pygame_gui.elements.UILabel(
            pygame.Rect(10, y, width - 20, 20),
            f"Speed: {C.TIME_STEP_BASE:.0f}",
            manager,
            container=self.panel,
        )
        y += 20
        self.speed_slider = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect(10, y, width - 20, 20),
            start_value=C.TIME_STEP_BASE,
            value_range=(10, 3600),
            manager=manager,
            container=self.panel,
        )
        y += 30
        self.trail_label = pygame_gui.elements.UILabel(
            pygame.Rect(10, y, width - 20, 20),
            f"Trail: {C.DEFAULT_TRAIL_LENGTH}",
            manager,
            container=self.panel,
        )
        y += 20
        self.trail_slider = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect(10, y, width - 20, 20),
            start_value=C.DEFAULT_TRAIL_LENGTH,
            value_range=(C.MIN_TRAIL_LENGTH, C.MAX_TRAIL_LENGTH),
            manager=manager,
            container=self.panel,
        )
        y += 30
        self.save_button = pygame_gui.elements.UIButton(
            pygame.Rect(10, y, 80, 25),
            "Save",
            manager,
            container=self.panel,
        )
        self.load_button = pygame_gui.elements.UIButton(
            pygame.Rect(100, y, 80, 25),
            "Load",
            manager,
            container=self.panel,
        )
        y += 35
        self.info_box = pygame_gui.elements.UITextBox(
            "",
            pygame.Rect(10, y, width - 20, 150),
            manager,
            container=self.panel,
        )

    def update_speed_label(self, value: float):
        self.speed_label.set_text(f"Speed: {value:.0f}")

    def update_trail_label(self, value: float):
        self.trail_label.set_text(f"Trail: {int(value)}")

    def update_body_info(self, body, central_body=None):
        if body is None:
            self.info_box.set_text("")
            return
        elems = calculate_orbital_elements(body, central_body) if central_body else {}
        text = (
            f"<b>{body.name}</b><br>"
            f"m={body.mass:.2e} kg<br>"
            f"pos={body.pos}<br>"
            f"vel={body.vel}<br>"
            f"a={elems.get('semi_major_axis', 0):.2e} m "
            f"e={elems.get('eccentricity', 0):.3f}"
        )
        self.info_box.set_text(text)
