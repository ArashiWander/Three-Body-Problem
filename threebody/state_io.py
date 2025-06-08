import json
from typing import List
from .rendering import Body


def save_state(filename: str, bodies: List[Body]):
    """Save simulation bodies to a JSON file."""
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
    with open(filename, "w") as f:
        json.dump(data, f)


def load_state(filename: str) -> List[Body]:
    """Load bodies from a JSON file."""
    with open(filename) as f:
        data = json.load(f)
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
