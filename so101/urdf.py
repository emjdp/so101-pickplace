"""URDF 에서 관절 정보만 읽는다. (XML 파싱은 실습 대상이 아니라 미리 만들어 둠)

URDF 의 <joint> 하나는 "부모 링크 좌표계에서 본 자식 링크 좌표계" 를 정의한다.
  <origin xyz="..." rpy="..."/>  관절각이 0 일 때 자식 좌표계의 위치와 자세
  <axis xyz="0 0 1"/>            관절이 도는 축 (자식 좌표계 기준)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


@dataclass
class Joint:
    name: str
    type: str  # "revolute" | "fixed" | ...
    parent: str
    child: str
    xyz: np.ndarray  # (3,) [m]
    rpy: np.ndarray  # (3,) [rad]  roll, pitch, yaw
    axis: np.ndarray  # (3,)
    lower: float | None  # [rad]
    upper: float | None  # [rad]


def _vec(elem, attr, default):
    if elem is None or elem.get(attr) is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in elem.get(attr).split()], dtype=float)


def load_joints(path) -> dict[str, Joint]:
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        limit = j.find("limit")
        joints[j.get("name")] = Joint(
            name=j.get("name"),
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            xyz=_vec(origin, "xyz", [0, 0, 0]),
            rpy=_vec(origin, "rpy", [0, 0, 0]),
            axis=_vec(j.find("axis"), "xyz", [1, 0, 0]),
            lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
            upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
        )
    return joints


def chain(joints: dict[str, Joint], base="base_link", tip="gripper_frame_link") -> list[Joint]:
    """base 에서 tip 까지 이어지는 관절을 순서대로 돌려준다.

    tip 에서 시작해 부모 쪽으로 거슬러 올라간 뒤 뒤집는다.
    SO-101 이면 shoulder_pan → shoulder_lift → elbow_flex → wrist_flex → wrist_roll → gripper_frame_joint(fixed)
    """
    by_child = {j.child: j for j in joints.values()}
    out = []
    link = tip
    while link != base:
        if link not in by_child:
            raise ValueError(f"'{link}' 에서 '{base}' 로 가는 관절이 없음")
        j = by_child[link]
        out.append(j)
        link = j.parent
    return out[::-1]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    import config

    np.set_printoptions(precision=4, suppress=True)
    for j in chain(load_joints(config.URDF_PATH)):
        print(f"{j.name:20s} {j.type:9s} {j.parent:>15s} -> {j.child:20s} xyz={j.xyz} rpy={j.rpy} axis={j.axis}")
