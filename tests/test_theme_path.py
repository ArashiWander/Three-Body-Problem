import os
import pygame
from threebody import simulation_full


def test_simulation_full_theme_load(tmp_path, monkeypatch):
    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    monkeypatch.setitem(os.environ, "SDL_AUDIODRIVER", "dummy")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
        simulation_full.main()
    finally:
        os.chdir(cwd)
