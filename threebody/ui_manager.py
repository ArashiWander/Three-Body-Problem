import pygame
import pygame_gui

# Allow compatibility with pygame_gui versions using different checkbox names
if hasattr(pygame_gui.elements, "UICheckBox"):
    _UICheckBox = pygame_gui.elements.UICheckBox
elif hasattr(pygame_gui.elements, "UICheckbox"):  # pragma: no cover - older versions
    _UICheckBox = pygame_gui.elements.UICheckbox
else:  # pragma: no cover - unexpected version
    raise ImportError("UICheckBox class not found in pygame_gui.elements")

from . import constants as C
from .presets import PRESETS
from .analysis import calculate_orbital_elements

# --- KEPT FROM THE 'codex' BRANCH ---
class ControlPanel:
    """Helper to build and manage the on-screen control panel for presets and playback."""

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


# --- KEPT FROM THE 'main' BRANCH ---
class UIManager:
    """Create and manage all pygame_gui elements for simulation parameters."""

    def __init__(
        self,
        manager: pygame_gui.UIManager,
        integrator="Symplectic",
        adaptive=False,
        use_gr=False,
        show_field=False,
    ):
        self.manager = manager

        panel = pygame_gui.elements.UIPanel(
            pygame.Rect(
                0, 0, C.UI_SIDEBAR_WIDTH, C.HEIGHT - C.UI_BOTTOM_HEIGHT,
            ),
            manager=self.manager,
            object_id="#control_panel", # Consider a different ID to avoid style conflicts
        )

        pygame_gui.elements.UILabel(
            pygame.Rect(0, 0, panel.rect.width, 30),
            text="Settings", # Changed text to differentiate panels
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
        self.adaptive_box = _UICheckBox(
            pygame.Rect(10, 70, panel.rect.width - 20, 20),
            "Adaptive",
            manager=self.manager,
            container=panel,
            initial_state=adaptive,
        )
        self.gr_box = _UICheckBox(
            pygame.Rect(10, 95, panel.rect.width - 20, 20),
            "GR Correction",
            manager=self.manager,
            container=panel,
            initial_state=use_gr,
        )
        self.field_box = _UICheckBox(
            pygame.Rect(10, 120, panel.rect.width - 20, 20),
            "Show Field",
            manager=self.manager,
            container=panel,
            initial_state=show_field,
        )
        # Other settings sliders...
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
        return None

    def update(self, time_delta):
        self.manager.update(time_delta)

    def draw(self, surface):
        self.manager.draw_ui(surface)
