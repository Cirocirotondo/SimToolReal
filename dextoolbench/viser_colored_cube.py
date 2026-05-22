"""Colored cube rendering for Viser (URDF materials are not shown by ViserUrdf)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from dextoolbench.eval_env_config import CUBE_FIXED_SIZE

# Matches generate_objects._cuboid_per_face_color_visuals (RGBA 0–1 → 0–255)
CUBE_FACE_RGB = {
    "px": (255, 38, 38),
    "nx": (255, 128, 0),
    "py": (38, 217, 38),
    "ny": (25, 140, 115),
    "pz": (51, 89, 255),
    "nz": (255, 217, 25),
}


def add_colored_cube_viser(
    server,
    parent_path: str,
    dyn_handles: List,
    scale: Sequence[float] = CUBE_FIXED_SIZE,
    opacity: Optional[float] = None,
) -> None:
    """Add six colored face boxes as children of a viser frame (local coordinates)."""
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    thickness = max(0.002, min(sx, sy, sz) * 0.04)
    t2 = thickness / 2

    faces = [
        ("px", (hx - t2, 0.0, 0.0), (thickness, sy, sz)),
        ("nx", (-hx + t2, 0.0, 0.0), (thickness, sy, sz)),
        ("py", (0.0, hy - t2, 0.0), (sx, thickness, sz)),
        ("ny", (0.0, -hy + t2, 0.0), (sx, thickness, sz)),
        ("pz", (0.0, 0.0, hz - t2), (sx, sy, thickness)),
        ("nz", (0.0, 0.0, -hz + t2), (sx, sy, thickness)),
    ]

    for name, position, dimensions in faces:
        kwargs = dict(
            color=CUBE_FACE_RGB[name],
            dimensions=dimensions,
            position=position,
            side="double",
        )
        if opacity is not None:
            kwargs["opacity"] = opacity
        handle = server.scene.add_box(f"{parent_path}/face_{name}", **kwargs)
        dyn_handles.append(handle)
