"""시스템 A 픽앤플레이스(so101/pickplace.py) 및 실험 집계(so101/experiment.py) 통합 테스트."""

import numpy as np
import pytest

import config
from so101.detect import Detection
from so101.experiment import TrialRecord, append_trial, format_markdown_report, load_trials, summarize_trials
from so101.fk import SO101Kinematics
from so101.homography import HomographyCalib, TopViewRayCalib
from so101.pickplace import (
    DryRunRobot,
    WorkspaceConfig,
    execute_pick_and_place,
    min_jerk,
    plan_min_jerk_trajectory,
)

if not config.URDF_PATH.exists():
    pytest.skip(f"URDF 없음: {config.URDF_PATH}", allow_module_level=True)


def test_min_jerk_boundary_conditions():
    assert min_jerk(0.0) == 0.0
    assert min_jerk(1.0) == 1.0
    assert np.isclose(min_jerk(0.5), 0.5)
    start = {"shoulder_pan": 0.0, "shoulder_lift": -50.0, "elbow_flex": 40.0, "wrist_flex": 10.0, "wrist_roll": 0.0, "gripper": 10.0}
    goal = {**start, "shoulder_pan": 30.0, "gripper": 100.0}
    traj = plan_min_jerk_trajectory(start, goal, speed_deg_s=60.0, rate_hz=30.0)
    assert len(traj) >= 2
    np.testing.assert_allclose(traj[-1][1]["shoulder_pan"], 30.0)
    np.testing.assert_allclose(traj[-1][1]["gripper"], 100.0)


def test_system_a_end_to_end_dry_run():
    """검출 -> 호모그래피 변환 -> IK -> 접근/하강/파지/상자이동/놓기/복귀 전체 흐름 검증."""
    kin = SO101Kinematics()
    ray = TopViewRayCalib.from_files(config.CAMERA_INTRINSICS, config.CAMERA_EXTRINSICS)
    calib = ray.to_homography(plane_z_m=0.025)
    ws = WorkspaceConfig()

    # 작업 영역 안의 블록 (0.24, 0.05)
    u, v = calib.xy_to_pixel(0.24, 0.05)
    det = Detection(u=u, v=v, angle_deg=20.0, area=900.0, box=np.zeros((4, 2)))
    robot = DryRunRobot(simulated_grip_opening=22.0)

    res = execute_pick_and_place(robot, kin, calib, det, ws=ws, realtime=False)
    assert res.ok
    assert res.failure_cause == ""
    assert abs(res.target_x - 0.24) < 0.002 and abs(res.target_y - 0.05) < 0.002
    assert len(robot.history) > 20

    # 미검출 시
    res_none = execute_pick_and_place(robot, kin, calib, None, ws=ws, realtime=False)
    assert not res_none.ok and res_none.failure_cause == "detection_fail"

    # 빈손(파지 실패) 시
    robot_empty = DryRunRobot(simulated_grip_opening=0.5)
    res_empty = execute_pick_and_place(robot_empty, kin, calib, det, ws=ws, realtime=False)
    assert not res_empty.ok and res_empty.failure_cause == "grasp_fail"


def test_experiment_logging_and_summary(tmp_path):
    csv_file = tmp_path / "trials.csv"
    append_trial(csv_file, TrialRecord("", "A", "base", 0, 1, 10.5))
    append_trial(csv_file, TrialRecord("", "A", "base", 0, 0, 12.0, failure_cause="coord_error"))
    append_trial(csv_file, TrialRecord("", "B", "base", 0, 1, 14.2))

    loaded = load_trials(csv_file)
    assert len(loaded) == 3
    assert loaded[1].trial == 2

    rows = summarize_trials(loaded)
    assert len(rows) == 2
    a_base = next(r for r in rows if r["system"] == "A")
    assert a_base["n_trials"] == 2 and a_base["n_success"] == 1 and a_base["success_rate_pct"] == 50.0
    md = format_markdown_report(loaded)
    assert "시스템 A" in md and "시스템 B" in md
