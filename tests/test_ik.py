"""역기구학(so101/ik.py) 테스트: 닫힌 해(2R 평면 기구학), 수치 해(LM), 하이브리드 해 검증."""

import math

import numpy as np
import pytest

import config
from so101.fk import SO101Kinematics
from so101.ik import CLOSING_AXIS, ik_analytic, ik_numerical, roll_for_block_angle, solve_grasp_ik

if not config.URDF_PATH.exists():
    pytest.skip(f"URDF 없음: {config.URDF_PATH}", allow_module_level=True)


@pytest.fixture(scope="module")
def kin():
    return SO101Kinematics()


@pytest.mark.parametrize(
    "target_xyz",
    [
        [0.20, 0.00, 0.02],
        [0.24, 0.08, 0.03],
        [0.22, -0.10, 0.04],
        [0.18, 0.05, 0.06],
    ],
)
def test_analytic_ik_top_down_matches_fk(kin, target_xyz):
    """닫힌 해로 구한 관절각을 FK에 넣으면 목표 끝점 위치가 0.05 mm 이내로 일치하고 아래(-z)를 향한다."""
    q = ik_analytic(kin, target_xyz, pitch_deg=90.0, roll_rad=0.2)
    assert q is not None
    T = kin.fk(q)
    np.testing.assert_allclose(T[:3, 3], target_xyz, atol=5e-5)
    assert T[2, 2] < -0.99
    assert np.all(q >= kin.lower) and np.all(q <= kin.upper)


def test_numerical_ik_random_roundtrip(kin):
    """임의 자세 40개에서 끝점 위치와 접근 방향으로부터 수치 IK가 1.5 mm 이내로 해를 복원한다."""
    rng = np.random.default_rng(7)
    solved = 0
    for _ in range(40):
        q_true = rng.uniform(kin.lower * 0.7, kin.upper * 0.7)
        q_true[4] = 0.0
        pos = kin.tip(q_true)
        app = kin.approach(q_true)
        q_sol = ik_numerical(kin, pos, approach=app, roll_rad=0.0)
        if q_sol is None:
            continue
        solved += 1
        assert np.linalg.norm(kin.tip(q_sol) - pos) < 0.0015
    assert solved >= 34


def test_solve_grasp_ik_tilts_for_far_targets(kin):
    """수직 하강(90°)으로는 닿지 않는 먼 작업점(x=0.33 m)도 기울임 허용 시 해를 찾는다."""
    far_pos = [0.33, 0.05, 0.02]
    assert ik_analytic(kin, far_pos, pitch_deg=90.0) is None
    q = solve_grasp_ik(kin, far_pos, roll_rad=0.0, allow_tilt=True, method="hybrid")
    assert q is not None
    assert np.linalg.norm(kin.tip(q) - far_pos) < 0.0015


def test_unreachable_returns_none(kin):
    assert ik_analytic(kin, [0.85, 0.0, 0.0]) is None
    assert solve_grasp_ik(kin, [0.85, 0.0, 0.0]) is None


def test_roll_for_block_angle_aligns_closing_axis(kin):
    """블록 회전각에 맞춰 고른 wrist_roll은 집게가 다무는 방향과 블록 축이 수직에 가깝게 정렬된다."""
    pos = [0.22, 0.04, 0.02]
    block_angle = math.radians(35.0)
    roll = roll_for_block_angle(kin, pos, block_angle)
    q = solve_grasp_ik(kin, pos, roll_rad=roll)
    assert q is not None
    closing_xy = kin.fk(q)[:2, CLOSING_AXIS]
    closing_xy /= np.linalg.norm(closing_xy)
    axis_xy = np.array([math.cos(block_angle), math.sin(block_angle)])
    assert abs(float(closing_xy @ axis_xy)) < 0.15
