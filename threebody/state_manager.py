from pathlib import Path
import json
from typing import List
from .rendering import Body


def save_state(filepath: str, bodies: List[Body]):
    """Serialize bodies to a JSON file."""
    data = []
    for b in bodies:
        data.append({
            "mass": b.mass,
            "pos": b.pos.tolist(),
            "vel": b.vel.tolist(),
            "color": list(b.color),
            "radius": int(getattr(b, "radius_pixels", 5)),
            "fixed": bool(b.fixed),
            "name": b.name,
        })
    Path(filepath).write_text(json.dumps(data))


def load_state(filepath: str) -> List[Body]:
    """Load bodies from a JSON file."""
    data = json.loads(Path(filepath).read_text())
    bodies = []
    for item in data:
        bodies.append(
            Body(
                item.get("mass", 0.0),
                item.get("pos", [0, 0, 0]),
                item.get("vel", [0, 0, 0]),
                tuple(item.get("color", [255, 255, 255])),
                item.get("radius", 5),
                fixed=item.get("fixed", False),
                name=item.get("name"),
            )
        )
    return bodies
