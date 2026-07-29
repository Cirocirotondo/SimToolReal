#!/usr/bin/env python3
"""Generate combined UR5e + Delto hand URDFs from the vendor hand assets."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
URDF_ROOT = HERE.parent
DG_DESCRIPTION = URDF_ROOT / "delto_m_ros2" / "dg_description"
LEFT_COMBINED = HERE / "ur5e_left_dg5f.urdf"
PACKAGE_PREFIX = "package://dg_description/"


def _is_left_hand_element(element: ET.Element) -> bool:
    name = element.get("name", "")
    return name.startswith(("ll_dg_", "lj_dg_")) or name == "ur5e_dg5f_mount"


def _indent(element: ET.Element, level: int = 0) -> None:
    """Python 3.8-compatible equivalent of ElementTree.indent."""
    whitespace = "\n" + level * "  "
    child_whitespace = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_whitespace
        for child in element:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_whitespace
        child.tail = whitespace
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = whitespace


def _correct_right_hand_limits(
    robot: ET.Element, corrected_left_limits: dict[str, tuple[str, str]]
) -> None:
    """Mirror the physically corrected limits from the combined left URDF."""
    for finger in range(1, 6):
        for joint_index in range(1, 5):
            suffix = f"dg_{finger}_{joint_index}"
            right_joint = robot.find(f"joint[@name='rj_{suffix}']")
            if right_joint is None:
                raise ValueError(f"Missing right-hand joint rj_{suffix}")

            limit = right_joint.find("limit")
            axis = right_joint.find("axis")
            if limit is None or axis is None:
                raise ValueError(f"Joint rj_{suffix} is missing limit or axis")

            left_lower, left_upper = corrected_left_limits[f"lj_{suffix}"]
            axis_xyz = tuple(float(value) for value in axis.get("xyz").split())
            if axis_xyz in {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}:
                limit.set("lower", str(-float(left_upper)))
                limit.set("upper", str(-float(left_lower)))
            else:
                limit.set("lower", left_lower)
                limit.set("upper", left_upper)


def generate(
    side: str,
    *,
    mount_yaw_deg: float = 0.0,
    output: Optional[Path] = None,
) -> Path:
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    tree = ET.parse(LEFT_COMBINED)
    robot = tree.getroot()
    robot.set("name", f"ur5e_{side}_dg5f")

    corrected_left_limits = {
        joint.get("name"): (
            joint.find("limit").get("lower"),
            joint.find("limit").get("upper"),
        )
        for joint in robot.findall("joint")
        if joint.get("name", "").startswith("lj_dg_")
        and joint.find("limit") is not None
    }

    for element in list(robot):
        if element.tag in {"link", "joint"} and _is_left_hand_element(element):
            robot.remove(element)

    hand_root = ET.parse(DG_DESCRIPTION / "urdf" / f"dg5f_{side}.urdf").getroot()
    # Append after the arm elements so Isaac Gym keeps the established
    # 6 arm DOFs + 20 hand DOFs ordering used by observations and actions.
    insert_at = len(robot)

    for element in hand_root:
        element = copy.deepcopy(element)
        for mesh in element.iter("mesh"):
            filename = mesh.get("filename")
            if filename and filename.startswith(PACKAGE_PREFIX):
                mesh.set(
                    "filename",
                    "urdf/delto_m_ros2/dg_description/"
                    + filename[len(PACKAGE_PREFIX) :],
                )
        robot.insert(insert_at, element)
        insert_at += 1

    if side == "right":
        _correct_right_hand_limits(robot, corrected_left_limits)

    link_prefix = "ll" if side == "left" else "rl"
    mount = ET.Element("joint", {"name": "ur5e_dg5f_mount", "type": "fixed"})
    mount_yaw_rad = math.radians(float(mount_yaw_deg))
    ET.SubElement(
        mount,
        "origin",
        {"xyz": "0 0 0", "rpy": f"0 0 {mount_yaw_rad:.15g}"},
    )
    ET.SubElement(mount, "parent", {"link": "wrist_3_link"})
    ET.SubElement(mount, "child", {"link": f"{link_prefix}_dg_mount"})
    robot.insert(insert_at, mount)

    _indent(robot)
    if output is None:
        output = HERE / f"ur5e_{side}_dg5f.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    return output


if __name__ == "__main__":
    generated_default = generate("right")
    generated_motion_imitation = generate(
        "right",
        mount_yaw_deg=60.0,
        output=HERE / "ur5e_right_dg5f_mount_60deg.urdf",
    )
    print(generated_default)
    print(generated_motion_imitation)
