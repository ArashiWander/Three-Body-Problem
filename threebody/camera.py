import numpy as np
from . import constants as C
from .physics_utils import calculate_center_of_mass


class Camera:
    """Manage view transformation and focus handling."""

    def __init__(self, zoom=C.ZOOM_BASE, pan_offset=None):
        self.zoom = float(zoom)
        self.pan_offset = (
            np.array(pan_offset, dtype=float) if pan_offset is not None else C.INITIAL_PAN_OFFSET.astype(float).copy()
        )

    def world_to_screen(self, pos):
        """Convert a world position to screen coordinates."""
        pos = np.asarray(pos, dtype=float)
        return pos[:2] * self.zoom + self.pan_offset

    def update_focus(self, focus_body, bodies, screen_center):
        """Smoothly update the camera pan to follow the focus target."""
        if focus_body is None:
            return
        if focus_body == "COM":
            com_pos, _ = calculate_center_of_mass(bodies)
            if com_pos is None:
                return
            target_pos = com_pos[:2]
        else:
            target_pos = focus_body.pos[:2]

        target = screen_center - target_pos * self.zoom
        self.pan_offset += (target - self.pan_offset) * C.CAMERA_SMOOTHING
