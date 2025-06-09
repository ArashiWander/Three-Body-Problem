import importlib
import types
import sys

import pygame_gui


def test_uicheckbox_import_fallback(monkeypatch):
    # create fake elements module without checkbox attributes
    fake_elements = types.SimpleNamespace(ui_check_box=pygame_gui.elements.ui_check_box)
    monkeypatch.setattr(pygame_gui, "elements", fake_elements)
    monkeypatch.setitem(sys.modules, "pygame_gui.elements", fake_elements)

    import threebody.ui_manager as ui_manager
    importlib.reload(ui_manager)

    assert ui_manager._UICheckBox is pygame_gui.elements.ui_check_box.UICheckBox

