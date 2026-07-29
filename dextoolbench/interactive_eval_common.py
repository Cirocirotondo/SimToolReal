"""Small shared utilities for interactive policy evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


def install_path_is_relative_to_backport() -> None:
    """Provide Path.is_relative_to() for the Python 3.8 Viser environment."""
    if hasattr(Path, "is_relative_to"):
        return

    def _is_relative_to(self: Path, *other: Path) -> bool:
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    Path.is_relative_to = _is_relative_to  # type: ignore[attr-defined]


def quat_xyzw_to_wxyz(quaternion) -> Tuple[float, float, float, float]:
    """Convert an Isaac Gym xyzw quaternion to Viser's wxyz ordering."""
    return (
        float(quaternion[3]),
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
    )


def checkpoint_payload(checkpoint: Any) -> Dict[str, Any]:
    """Return the single-policy payload used by SimToolReal checkpoints."""
    if isinstance(checkpoint, dict) and 0 in checkpoint:
        checkpoint = checkpoint[0]
    elif isinstance(checkpoint, (list, tuple)):
        if not checkpoint:
            raise ValueError("Checkpoint policy list is empty")
        checkpoint = checkpoint[0]
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint payload to be a dict, got {type(checkpoint)}"
        )
    return checkpoint
