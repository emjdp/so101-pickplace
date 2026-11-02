import numpy as np
import pytest

import config
from so101 import fk as fkmod

if not config.URDF_PATH.exists():
    pytest.skip(
        f"URDF 없음: {config.URDF_PATH}\n"
        "  git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100",
        allow_module_level=True,
    )

from so101.fk import ARM_JOINTS, SO101Kinematics, joint_transform  # noqa: E402

PAN_AXIS_XY = np.array([0.0388353, 0.0])  # URDF shoulder_pan origin (base_link 기준)


@pytest.fixture(scope="module")
def kin():
    return SO101Kinematics()


def test_fixed_joint_ignores_angle(kin):
    tool_joint = kin.joints[-1]
    assert tool_joint.type == "fixed"
    np.testing.assert_allclose(joint_transform(tool_joint, 0.0), joint_transform(tool_joint, 1.0))


def test_zero_pose_reference(kin):
    """관절각이 모두 0 일 때 그리퍼 끝 위치.

    기준값은 Claude 가 so101_new_calib.urdf (SO-ARM100 main, 2026-11-02) 로 계산한 값이다.
    URDF 가 업데이트돼서 값이 바뀌었을 수도 있으니, 틀리면 yourdfpy 비교 테스트 결과도 같이 볼 것.
    """
    T = kin.fk(np.zeros(5))
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, 3], [0.3914, 0.0, 0.2265], atol=1e-3)


def test_pan_turns_about_vertical_axis(kin):
    """shoulder_pan 만 돌리면 그리퍼 끝은 pan 축 둘레의 원 위를 움직인다 (높이, 축까지 거리 불변)."""
    base = kin.fk([0, 0.3, -0.4, 0.2, 0])[:3, 3]
    for pan in [-1.0, -0.3, 0.5, 1.2]:
        p = kin.fk([pan, 0.3, -0.4, 0.2, 0])[:3, 3]
        assert np.isclose(p[2], base[2], atol=1e-9)
        assert np.isclose(np.linalg.norm(p[:2] - PAN_AXIS_XY), np.linalg.norm(base[:2] - PAN_AXIS_XY), atol=1e-9)


def test_wrist_flex_90_points_down(kin):
    """wrist_flex +90° 이면 그리퍼 접근 방향(tool z 축)이 바닥(-z)을 향한다."""
    T = kin.fk(np.deg2rad([0, 0, 0, 90, 0]))
    np.testing.assert_allclose(T[:3, 2], [0, 0, -1], atol=1e-3)


def test_lerobot_conversion_uses_sign_and_offset():
    joints_deg = {"shoulder_pan": 10.0, "shoulder_lift": -20.0, "elbow_flex": 30.0, "wrist_flex": 40.0, "wrist_roll": -50.0, "gripper": 12.0}
    expected = [np.deg2rad(fkmod.JOINT_SIGN[n] * (joints_deg[n] - fkmod.JOINT_OFFSET_DEG[n])) for n in ARM_JOINTS]
    np.testing.assert_allclose(fkmod.lerobot_to_urdf_rad(joints_deg), expected)


def test_matches_yourdfpy(kin):
    """독립 구현(yourdfpy)과 임의 자세 20개에서 비교. pip install yourdfpy 가 없으면 건너뛴다."""
    yourdfpy = pytest.importorskip("yourdfpy")
    robot = yourdfpy.URDF.load(
        str(config.URDF_PATH), load_meshes=False, build_collision_scene_graph=False, load_collision_meshes=False
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, size=5)
        robot.update_cfg(dict(zip(ARM_JOINTS, q)))
        expected = robot.get_transform("gripper_frame_link", "base_link")
        np.testing.assert_allclose(kin.fk(q), expected, atol=1e-6)
