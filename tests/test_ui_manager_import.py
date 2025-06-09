import importlib
import sys
import types
import pygame_gui


def test_uicheckbox_import_fallback(monkeypatch):
    """Ensure UICheckBox can be imported from a submodule."""
    fake_module = types.ModuleType("pygame_gui.elements.ui_check_box")

    class DummyCheckBox:  # noqa: D401 - simple dummy class
        """Dummy checkbox class."""

    fake_module.UICheckBox = DummyCheckBox
    fake_elements = types.SimpleNamespace(ui_check_box=fake_module)
    monkeypatch.setattr(pygame_gui, "elements", fake_elements)
    monkeypatch.setitem(sys.modules, "pygame_gui.elements", fake_elements)
    monkeypatch.setitem(
        sys.modules,
        "pygame_gui.elements.ui_check_box",
        fake_module,
    )

    import threebody.ui_manager as ui_manager
    importlib.reload(ui_manager)
    assert ui_manager._UICheckBox is DummyCheckBox

