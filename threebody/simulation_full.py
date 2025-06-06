"""Legacy entry point for running the full simulation."""

try:
    import pygame_gui  # noqa: F401
    PYGAME_GUI_AVAILABLE = True
except ImportError:
    print("Error: pygame_gui not found. Please install it: pip install pygame_gui")
    PYGAME_GUI_AVAILABLE = False

from .simulation import Simulation


def main():
    """Launch the interactive simulation."""
    Simulation().run()


if __name__ == "__main__":
    if not PYGAME_GUI_AVAILABLE:
        print("\nExiting due to missing pygame_gui dependency.")
    else:
        main()
