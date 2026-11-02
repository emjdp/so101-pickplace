"""SO-101 순기구학(FK). ★ 실습: TODO 를 채운다 ★

확인: pytest tests/test_fk.py -v
      python scripts/fk_try.py 0 0 0 0 0 --plot
참고: 문서 '2. 순기구학 FK' 3~4절

흐름
  LeRobot 관절값 [deg]  --(부호·영점 변환)-->  URDF 관절각 [rad]  --(관절 변환 곱)-->  T_base_tool (4x4)
"""

import numpy as np

import config
from so101.transforms import make_transform, rot_z, rpy_to_matrix
from so101.urdf import Joint, chain, load_joints

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# ── LeRobot ↔ URDF 각도 맞추기 ─────────────────────────────────────
#   q_urdf[deg] = JOINT_SIGN * (q_lerobot[deg] - JOINT_OFFSET_DEG)
# LeRobot 은 so101_new_calib.urdf 와 부호·영점이 같다고 보고 deg2rad 만 해서 넘긴다
# (lerobot/model/kinematics.py). 그래서 처음 값은 +1, 0 이다.
# 실측해 보고 다르면 여기를 고치고, 고친 이유를 문서 '4. 측정 기록' 에 남긴다.
JOINT_SIGN = {name: 1.0 for name in ARM_JOINTS}
JOINT_OFFSET_DEG = {name: 0.0 for name in ARM_JOINTS}


def lerobot_to_urdf_rad(joints_deg: dict) -> np.ndarray:
    """{관절 이름: LeRobot 각도[deg]} → URDF 관절각 배열 (5,) [rad], ARM_JOINTS 순서.

    joints_deg 에는 "shoulder_pan" 같은 이름이 키로 들어온다 (gripper 가 섞여 있어도 무시).
    """
    raise NotImplementedError("TODO: lerobot_to_urdf_rad")


def joint_transform(joint: Joint, q: float) -> np.ndarray:
    """관절 하나의 변환 T_parent_child(q).

    1) origin: 관절각 0 일 때 부모 → 자식 좌표계 변환 (joint.xyz, joint.rpy)
    2) 관절 회전: 자식 좌표계에서 joint.axis 를 축으로 q 만큼 회전
    T = T_origin @ T_rotation   (곱하는 순서가 왜 이렇게 되는지 문서에서 확인)

    - fixed 관절이면 q 를 무시하고 origin 만 돌려준다.
    - SO-101 URDF 의 회전 관절 축은 전부 z (0 0 1) 이다. 다른 축이 들어오면 에러를 내도 된다.
    """
    raise NotImplementedError("TODO: joint_transform")


def forward_kinematics(joints: list[Joint], q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """관절 체인과 관절각으로 T_base_tool 을 구한다.

    joints: urdf.chain() 결과 (회전 관절 5개 + 마지막 fixed 1개)
    q:      회전 관절 각도 (5,) [rad]. fixed 관절에는 각도가 없다는 점에 주의.
    반환:   (T_base_tool, 각 관절을 지난 뒤의 누적 변환 리스트)
            누적 리스트는 fk_try.py 의 막대 그림에 쓴다.
    """
    raise NotImplementedError("TODO: forward_kinematics")


class SO101Kinematics:
    """URDF 를 한 번 읽어 두고 FK 를 계산한다. (스크립트에서 쓰는 창구)"""

    def __init__(self, urdf_path=config.URDF_PATH):
        self.joints = chain(load_joints(urdf_path))
        revolute = [j.name for j in self.joints if j.type != "fixed"]
        assert revolute == ARM_JOINTS, f"URDF 관절 순서가 예상과 다름: {revolute}"

    def fk(self, q_rad) -> np.ndarray:
        T, _ = forward_kinematics(self.joints, np.asarray(q_rad, dtype=float))
        return T

    def fk_frames(self, q_rad) -> list[np.ndarray]:
        _, frames = forward_kinematics(self.joints, np.asarray(q_rad, dtype=float))
        return frames

    def fk_from_lerobot(self, joints_deg: dict) -> np.ndarray:
        return self.fk(lerobot_to_urdf_rad(joints_deg))
