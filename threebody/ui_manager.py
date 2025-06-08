from pathlib import Path
import pygame
import pygame_gui

from . import constants as C


class UIManager:
    """Create and manage all pygame_gui elements."""

    def __init__(
        self,
        integrator="Symplectic",
        adaptive=False,
        use_gr=False,
        show_field=False,
        *,
        theme_path=None,
    ):
        if theme_path is None:
            theme_path = Path(__file__).with_name("theme.json")
        self.manager = pygame_gui.UIManager((C.WIDTH, C.HEIGHT), theme_path)

        panel = pygame_gui.elements.UIPanel(
            pygame.Rect(
                C.WIDTH - C.UI_SIDEBAR_WIDTH,
                0,
                C.UI_SIDEBAR_WIDTH,
                C.HEIGHT - C.UI_BOTTOM_HEIGHT,
            ),
            manager=self.manager,
            object_id="#control_panel",
        )

        pygame_gui.elements.UILabel(
            pygame.Rect(0, 0, panel.rect.width, 30),
            text="Controls",
            manager=self.manager,
            container=panel,
            object_id="#title_label",
        )

        self.integrator_menu = pygame_gui.elements.UIDropDownMenu(
            ["Symplectic", "RK4"],
            integrator,
            pygame.Rect(10, 40, panel.rect.width - 20, 25),
            manager=self.manager,
            container=panel,
        )
        self.adaptive_box = pygame_gui.elements.UICheckBox(
            pygame.Rect(10, 70, panel.rect.width - 20, 20),
            "Adaptive",
            manager=self.manager,
            container=panel,
            initial_state=adaptive,
        )
        self.gr_box = pygame_gui.elements.UICheckBox(
            pygame.Rect(10, 95, panel.rect.width - 20, 20),
            "GR Correction",
            manager=self.manager,
            container=panel,
            initial_state=use_gr,
        )
        self.field_box = pygame_gui.elements.UICheckBox(
            pygame.Rect(10, 120, panel.rect.width - 20, 20),
            "Show Field",
            manager=self.manager,
            container=panel,
            initial_state=show_field,
        )

        self.ts_label = pygame_gui.elements.UILabel(
            pygame.Rect(10, 145, panel.rect.width - 20, 20),
            f"dt: {C.TIME_STEP_BASE:.0f}",
            manager=self.manager,
            container=panel,
        )
        self.ts_slider = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect(10, 165, panel.rect.width - 20, 20),
            start_value=C.TIME_STEP_BASE,
            value_range=(10, 3600),
            manager=self.manager,
            container=panel,
        )

        self.soft_label = pygame_gui.elements.UILabel(
            pygame.Rect(10, 190, panel.rect.width - 20, 20),
            f"soft: {C.SOFTENING_LENGTH:.2e}",
            manager=self.manager,
            container=panel,
        )
        self.soft_slider = pygame_gui.elements.UIHorizontalSlider(
            pygame.Rect(10, 210, panel.rect.width - 20, 20),
            start_value=C.SOFTENING_LENGTH,
            value_range=(0.1, 10.0),
            manager=self.manager,
            container=panel,
        )

        self.save_button = pygame_gui.elements.UIButton(
            pygame.Rect(10, 240, 60, 25),
            "Save",
            manager=self.manager,
            container=panel,
        )
        self.load_button = pygame_gui.elements.UIButton(
            pygame.Rect(80, 240, 60, 25),
            "Load",
            manager=self.manager,
            container=panel,
        )

        self.integrator = integrator
        self.adaptive = adaptive
        self.use_gr = use_gr
        self.show_field = show_field

    def process_event(self, event):
        self.manager.process_events(event)
        if (
            event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED
            and event.ui_element == self.integrator_menu
        ):
            self.integrator = event.text
        if event.type in (
            pygame_gui.UI_CHECK_BOX_CHECKED,
            pygame_gui.UI_CHECK_BOX_UNCHECKED,
        ):
            state = event.ui_element.checked
            if event.ui_element == self.adaptive_box:
                self.adaptive = state
            elif event.ui_element == self.gr_box:
                self.use_gr = state
            elif event.ui_element == self.field_box:
                self.show_field = state
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.ts_slider:
                C.TIME_STEP_BASE = event.value
                self.ts_label.set_text(f"dt: {event.value:.0f}")
            elif event.ui_element == self.soft_slider:
                C.SOFTENING_LENGTH = float(event.value)
                C.SOFTENING_FACTOR_SQ = C.SOFTENING_LENGTH ** 2
                self.soft_label.set_text(f"soft: {C.SOFTENING_LENGTH:.2e}")
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.save_button:
                return "save"
            if event.ui_element == self.load_button:
                return "load"
        return None

    def update(self, time_delta):
        self.manager.update(time_delta)

    def draw(self, surface):
        self.manager.draw_ui(surface)
