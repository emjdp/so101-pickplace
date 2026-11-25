"""시스템 A: 기구학 기반 픽앤플레이스 동작 순서 및 궤적 제어.

계획서 3.2절 시스템 A의 전체 처리 흐름:
  탑뷰 카메라 (C270)
    -> 블록 검출: HSV 색 범위로 영역 추출, 중심 픽셀 (u, v)와 회전각 계산 (`so101/detect.py`)
    -> 좌표 변환: 호모그래피로 (u, v) -> 로봇 베이스 기준 (x, y) (`so101/homography.py`)
    -> 역기구학: (x, y, z, 블록 회전각) -> 관절각 5개 (`so101/ik.py`)
    -> 동작 순서: 접근 -> 하강 -> 그리퍼 닫기 -> 상승 -> 상자 위로 이동 -> 그리퍼 열기 -> 복귀
    -> LeRobot 모터 명령 -> 팔로워암

동작 제어에 쓰는 기법:
- 관절 공간 최소 저크(minimum-jerk) 궤적 보간
- FK 기반 작업대·바닥 충돌 안전 검사 (`check_pose_safety`)
- 끝점(고정 집게) 비켜 잡기 오프셋 (`grasp_shift_m`): 물체가 두 집게 손가락 중앙에 오도록 보정
- 중력 처짐 보정 (`trim_gravity_sag`): 하강 후 실제 엔코더 관절각의 FK 끝점 오차를 읽어 한 번 더 보정
- 그리퍼 개도(opening %) 수렴 확인으로 실제 파지 성공 여부 판정
- 로봇·카메라 없이 전체 파이프라인을 검증할 수 있는 `DryRunRobot` 지원
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from so101.detect import Detection
from so101.fk import ARM_JOINTS, SO101Kinematics
from so101.homography import HomographyCalib
from so101.ik import CLOSING_AXIS, roll_for_block_angle, solve_grasp_ik
from so101.robot import ALL_MOTORS, joint_positions

DEFAULT_POSES: dict[str, dict[str, float]] = {
    "rest": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -104.0,
        "elbow_flex": 96.0,
        "wrist_flex": -100.0,
        "wrist_roll": 0.0,
    },
    "ready": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -80.0,
        "elbow_flex": 30.0,
        "wrist_flex": 80.0,
        "wrist_roll": 0.0,
    },
}


def min_jerk(s: float) -> float:
    """최소 저크 보간 다항식: s in [0, 1] -> [0, 1] (시작·끝 속도·가속도 0)."""
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def plan_min_jerk_trajectory(
    start_deg: dict[str, float],
    goal_deg: dict[str, float],
    speed_deg_s: float = 60.0,
    rate_hz: float = 40.0,
) -> list[tuple[float, dict[str, float]]]:
    """start_deg 에서 goal_deg 까지 최소 저크 보간 궤적 [(시각_s, {관절: deg}), ...] 을 생성한다."""
    full_goal = {**start_deg, **{m: float(v) for m, v in goal_deg.items() if m in ALL_MOTORS}}
    span = max((abs(full_goal[m] - start_deg[m]) for m in ARM_JOINTS), default=0.0)
    gspan = abs(full_goal.get("gripper", start_deg.get("gripper", 0.0)) - start_deg.get("gripper", 0.0))
    duration = max(1.875 * span / max(speed_deg_s, 1.0), gspan / 45.0, 0.25)
    n = max(2, int(round(duration * rate_hz)))
    steps = []
    for i in range(1, n + 1):
        s = min_jerk(i / n)
        pose = {m: float(start_deg[m] + s * (full_goal[m] - start_deg[m])) for m in ALL_MOTORS if m in start_deg}
        steps.append((i / rate_hz, pose))
    return steps


@dataclass
class WorkspaceConfig:
    """픽앤플레이스 작업 영역과 상자 위치, 높이 파라미터."""

    table_z_m: float = 0.0
    block_height_m: float = 0.025
    hover_m: float = 0.08
    grasp_clear_m: float = 0.008
    floor_clear_m: float = 0.002
    grasp_shift_m: float = 0.015
    grasp_min_opening: float = 4.0
    box_xyz_m: tuple[float, float, float] = (0.20, -0.15, 0.06)
    x_range_m: tuple[float, float] = (0.12, 0.38)
    y_range_m: tuple[float, float] = (-0.22, 0.22)
    speed_deg_s: float = 65.0
    slow_deg_s: float = 35.0
    rate_hz: float = 40.0

    def is_reachable_xy(self, x: float, y: float) -> bool:
        return self.x_range_m[0] <= x <= self.x_range_m[1] and self.y_range_m[0] <= y <= self.y_range_m[1]


@dataclass
class PickPlacePlan:
    """검출된 블록을 집어 상자에 넣기 위한 계산 결과 및 단계별 목표 관절각."""

    u: float
    v: float
    angle_deg: float
    block_x: float
    block_y: float
    grasp_x: float
    grasp_y: float
    grasp_z: float
    hover_z: float
    roll_rad: float
    above_deg: dict[str, float]
    down_deg: dict[str, float]
    box_above_deg: dict[str, float]
    box_release_deg: dict[str, float]


@dataclass
class PickPlaceResult:
    """시스템 A 1회 실행 결과 (실험 로그 기록용)."""

    ok: bool
    failure_cause: str = ""  # "", "detection_fail", "out_of_range", "ik_fail", "grasp_fail", "drop_fail"
    elapsed_s: float = 0.0
    u: float | None = None
    v: float | None = None
    angle_deg: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    gripper_opening: float | None = None
    sag_trim_mm: list[float] = field(default_factory=list)
    detail: str = ""


def check_pose_safety(
    kin: SO101Kinematics,
    joints_deg: dict[str, float],
    ws: WorkspaceConfig,
) -> str:
    """자세가 관절 한계를 벗어나거나 책상 바닥 아래로 내려가면 이유를 돌려주고, 안전하면 ''를 돌려준다."""
    q = kin.to_urdf(joints_deg)
    if np.any(q < kin.lower - 0.02) or np.any(q > kin.upper + 0.02):
        return "관절 가동 범위 초과"
    pts = kin.points(q)
    floor = ws.table_z_m + ws.floor_clear_m
    for name, p in zip(("pan", "lift", "elbow", "wrist", "roll", "tip"), pts):
        if name in ("pan", "lift"):
            continue
        if p[0] < -0.12:
            return f"{name} 관절이 베이스 뒤쪽으로 넘어감 (x={p[0]:.3f})"
        min_z = floor if name == "tip" else floor + 0.015
        if p[0] > 0.11 and p[2] < min_z - 1e-4:
            return f"{name} 높이가 너무 낮음 (z={p[2]:.3f} < {min_z:.3f})"
    return ""


def plan_pick_and_place(
    kin: SO101Kinematics,
    calib: HomographyCalib,
    det: Detection,
    ws: WorkspaceConfig,
    ik_method: str = "hybrid",
) -> tuple[PickPlacePlan | None, str]:
    """검출 결과 `det` 로부터 픽앤플레이스 경로의 모든 핵심 관절 자세를 계획한다.

    반환: (PickPlacePlan | None, 실패_원인_코드)
    """
    bx, by = calib.pixel_to_xy(det.u, det.v)
    if not ws.is_reachable_xy(bx, by):
        return None, "out_of_range"

    grasp_z = ws.table_z_m + ws.grasp_clear_m
    hover_z = ws.table_z_m + max(ws.block_height_m, 0.02) + ws.hover_m

    world_angle = calib.pixel_angle_to_world(det.u, det.v, det.angle_deg)
    roll = roll_for_block_angle(kin, [bx, by, grasp_z], world_angle)

    q_down0 = solve_grasp_ik(kin, [bx, by, grasp_z], roll_rad=roll, allow_tilt=True, method=ik_method)
    if q_down0 is None:
        return None, "ik_fail"

    gx, gy = bx, by
    if ws.grasp_shift_m:
        # 고정 집게 끝에서 두 집게 사이 중앙으로 물체가 오도록 다무는 축을 따라 이동
        closing = kin.fk(q_down0)[:3, CLOSING_AXIS]
        ax = closing[:2] / (np.linalg.norm(closing[:2]) + 1e-9)
        offset = min(ws.grasp_shift_m, ws.block_height_m * 0.5)
        gx, gy = float(bx + offset * ax[0]), float(by + offset * ax[1])

    q_down = solve_grasp_ik(kin, [gx, gy, grasp_z], roll_rad=roll, seed=q_down0, allow_tilt=True, method=ik_method)
    q_above = solve_grasp_ik(kin, [gx, gy, hover_z], roll_rad=roll, seed=q_down, allow_tilt=True, method=ik_method)
    if q_down is None or q_above is None:
        return None, "ik_fail"

    box_x, box_y, box_z = ws.box_xyz_m
    box_hover_z = max(hover_z, box_z + 0.05)
    q_box_above = solve_grasp_ik(kin, [box_x, box_y, box_hover_z], roll_rad=0.0, seed=q_above, allow_tilt=True, method=ik_method)
    q_box_rel = solve_grasp_ik(kin, [box_x, box_y, box_z], roll_rad=0.0, seed=q_box_above, allow_tilt=True, method=ik_method)
    if q_box_above is None or q_box_rel is None:
        return None, "ik_fail"

    down_deg = kin.to_lerobot(q_down)
    above_deg = kin.to_lerobot(q_above)
    box_above_deg = kin.to_lerobot(q_box_above)
    box_rel_deg = kin.to_lerobot(q_box_rel)

    for pose_deg in (above_deg, down_deg, box_above_deg, box_rel_deg):
        why = check_pose_safety(kin, pose_deg, ws)
        if why:
            return None, "ik_fail"

    return (
        PickPlacePlan(
            u=det.u,
            v=det.v,
            angle_deg=det.angle_deg,
            block_x=bx,
            block_y=by,
            grasp_x=gx,
            grasp_y=gy,
            grasp_z=grasp_z,
            hover_z=hover_z,
            roll_rad=roll,
            above_deg=above_deg,
            down_deg=down_deg,
            box_above_deg=box_above_deg,
            box_release_deg=box_rel_deg,
        ),
        "",
    )


class DryRunRobot:
    """실물 팔로워 없이 시스템 A 동작 순서와 궤적을 시험하기 위한 가상 로봇."""

    def __init__(self, initial_pose: dict[str, float] | None = None, simulated_grip_opening: float = 22.0):
        base = dict(DEFAULT_POSES["rest"], gripper=10.0)
        if initial_pose:
            base.update(initial_pose)
        self.pos = base
        self.simulated_grip_opening = simulated_grip_opening
        self.history: list[dict[str, float]] = []

    def get_observation(self) -> dict[str, float]:
        return {f"{m}.pos": float(self.pos[m]) for m in ALL_MOTORS}

    def send_action(self, action: dict[str, float]) -> None:
        for m in ALL_MOTORS:
            k = f"{m}.pos"
            if k in action:
                val = float(action[k])
                if m == "gripper" and val < self.simulated_grip_opening:
                    val = self.simulated_grip_opening
                self.pos[m] = val
        self.history.append(dict(self.pos))

    def disconnect(self) -> None:
        pass


def move_smooth(
    robot,
    goal_deg: dict[str, float],
    speed_deg_s: float = 60.0,
    rate_hz: float = 40.0,
    realtime: bool = True,
) -> dict[str, float]:
    """현재 관절 위치에서 goal_deg 까지 최소 저크 궤적으로 부드럽게 이동한다."""
    start = joint_positions(robot.get_observation())
    steps = plan_min_jerk_trajectory(start, goal_deg, speed_deg_s=speed_deg_s, rate_hz=rate_hz)
    t0 = time.perf_counter()
    for t_step, pose in steps:
        robot.send_action({f"{m}.pos": pose[m] for m in ALL_MOTORS})
        if realtime:
            wait = (t0 + t_step) - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
    return joint_positions(robot.get_observation())


def wait_gripper_settled(
    robot,
    timeout_s: float = 1.5,
    realtime: bool = True,
    target_gripper: float | None = None,
) -> float:
    """집게가 목표 개도에 도달하거나 물체에 걸려 멈출 때까지 기다린 뒤 개도(%)를 반환한다."""
    if not realtime:
        return joint_positions(robot.get_observation())["gripper"]
    end = time.perf_counter() + timeout_s
    hold_obs = joint_positions(robot.get_observation())
    hold_arm = {f"{m}.pos": hold_obs[m] for m in ARM_JOINTS}
    last = hold_obs["gripper"]
    still = 0
    while time.perf_counter() < end and still < 5:
        if target_gripper is not None:
            robot.send_action({**hold_arm, "gripper.pos": float(target_gripper)})
        time.sleep(0.06)
        now = joint_positions(robot.get_observation())["gripper"]
        still = still + 1 if abs(now - last) < 0.3 else 0
        last = now
    return last


def trim_gravity_sag(
    robot,
    kin: SO101Kinematics,
    target_pos,
    roll_rad: float,
    ws: WorkspaceConfig,
    realtime: bool = True,
) -> list[float]:
    """하강 후 중력으로 처진 끝점 오차(5~9 mm)를 실제 엔코더 FK로 읽어 한 번 더 보정한다."""
    if realtime:
        time.sleep(0.2)
    cur_deg = joint_positions(robot.get_observation())
    cur_tip = kin.tip(kin.to_urdf(cur_deg))
    err = np.asarray(target_pos, dtype=float) - cur_tip
    err_norm = float(np.linalg.norm(err))
    if not (0.002 < err_norm < 0.030):
        return [0.0, 0.0, 0.0]
    q_fix = solve_grasp_ik(
        kin,
        np.asarray(target_pos, dtype=float) + err,
        roll_rad=roll_rad,
        seed=kin.to_urdf(cur_deg),
        allow_tilt=True,
    )
    if q_fix is None:
        return [0.0, 0.0, 0.0]
    fix_deg = kin.to_lerobot(q_fix)
    if check_pose_safety(kin, fix_deg, ws):
        return [0.0, 0.0, 0.0]
    move_smooth(robot, fix_deg, speed_deg_s=ws.slow_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    return [round(float(v * 1000), 1) for v in err]


def execute_pick_and_place(
    robot,
    kin: SO101Kinematics,
    calib: HomographyCalib,
    det: Detection | None,
    ws: WorkspaceConfig | None = None,
    ik_method: str = "hybrid",
    realtime: bool = True,
) -> PickPlaceResult:
    """계획서 3.2절 시스템 A의 전체 픽앤플레이스 동작을 1회 수행하고 결과를 반환한다.

    동작 순서:
      1) 검출 확인 및 호모그래피 좌표 변환 → 역기구학 경로 계획
      2) 준비 자세(ready) + 그리퍼 열기(100%)
      3) 블록 위 접근(above) → 수직 하강(down) → 중력 처짐 보정
      4) 그리퍼 닫기(0%) → 개도로 파지 확인 → 상승(above)
      5) 상자 위로 이동(box_above) → 상자 위치(box_release) → 그리퍼 열기(100%)
      6) 상자 위 상승 → 휴식 자세(rest) 복귀
    """
    ws = ws or WorkspaceConfig()
    t0 = time.perf_counter()

    if det is None:
        return PickPlaceResult(ok=False, failure_cause="detection_fail", detail="블록 미검출")

    plan, fail_code = plan_pick_and_place(kin, calib, det, ws, ik_method=ik_method)
    if plan is None:
        bx, by = calib.pixel_to_xy(det.u, det.v)
        return PickPlaceResult(
            ok=False,
            failure_cause=fail_code,
            elapsed_s=round(time.perf_counter() - t0, 2),
            u=det.u,
            v=det.v,
            angle_deg=det.angle_deg,
            target_x=round(bx, 4),
            target_y=round(by, 4),
            detail=f"계획 실패 ({fail_code}): ({bx:.3f}, {by:.3f}) m",
        )

    # 1. 준비 자세 및 그리퍼 열기 (75%로 열어 불필요한 충돌·클램핑 방지)
    move_smooth(robot, {**DEFAULT_POSES["ready"], "gripper": 75.0}, speed_deg_s=ws.speed_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    # 2. 블록 위 접근
    move_smooth(robot, {**plan.above_deg, "gripper": 75.0}, speed_deg_s=ws.speed_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    # 3. 하강 및 처짐 보정
    move_smooth(robot, {**plan.down_deg, "gripper": 75.0}, speed_deg_s=ws.slow_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    sag_mm = trim_gravity_sag(robot, kin, [plan.grasp_x, plan.grasp_y, plan.grasp_z], plan.roll_rad, ws, realtime=realtime)

    # 4. 그리퍼 닫기 및 파지 확인
    move_smooth(robot, {"gripper": 0.0}, speed_deg_s=ws.slow_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    grip = wait_gripper_settled(robot, realtime=realtime, target_gripper=0.0)
    held = grip > ws.grasp_min_opening

    # 5. 상승
    move_smooth(robot, plan.above_deg, speed_deg_s=ws.slow_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    if not held:
        move_smooth(robot, {**DEFAULT_POSES["rest"], "gripper": 30.0}, speed_deg_s=ws.speed_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
        return PickPlaceResult(
            ok=False,
            failure_cause="grasp_fail",
            elapsed_s=round(time.perf_counter() - t0, 2),
            u=plan.u,
            v=plan.v,
            angle_deg=plan.angle_deg,
            target_x=round(plan.block_x, 4),
            target_y=round(plan.block_y, 4),
            gripper_opening=round(grip, 1),
            sag_trim_mm=sag_mm,
            detail=f"집게 빈손 닫힘 (gripper={grip:.1f}% <= {ws.grasp_min_opening}%)",
        )

    # 6. 상자 위로 이동 중 놓침 확인
    move_smooth(robot, plan.box_above_deg, speed_deg_s=ws.slow_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    grip_carry = wait_gripper_settled(robot, timeout_s=0.4, realtime=realtime)
    if grip_carry <= ws.grasp_min_opening:
        move_smooth(robot, {**DEFAULT_POSES["rest"], "gripper": 30.0}, speed_deg_s=ws.speed_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
        return PickPlaceResult(
            ok=False,
            failure_cause="drop_fail",
            elapsed_s=round(time.perf_counter() - t0, 2),
            u=plan.u,
            v=plan.v,
            angle_deg=plan.angle_deg,
            target_x=round(plan.block_x, 4),
            target_y=round(plan.block_y, 4),
            gripper_opening=round(grip_carry, 1),
            sag_trim_mm=sag_mm,
            detail="이송 중 블록 낙하",
        )

    # 7. 상자 위에서 그리퍼 열기 후 복귀
    move_smooth(robot, plan.box_release_deg, speed_deg_s=ws.slow_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    move_smooth(robot, {"gripper": 75.0}, speed_deg_s=ws.speed_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    wait_gripper_settled(robot, timeout_s=0.6, realtime=realtime, target_gripper=75.0)
    if realtime:
        time.sleep(0.2)
    move_smooth(robot, plan.box_above_deg, speed_deg_s=ws.speed_deg_s, rate_hz=ws.rate_hz, realtime=realtime)
    move_smooth(robot, {**DEFAULT_POSES["rest"], "gripper": 15.0}, speed_deg_s=ws.speed_deg_s, rate_hz=ws.rate_hz, realtime=realtime)

    return PickPlaceResult(
        ok=True,
        failure_cause="",
        elapsed_s=round(time.perf_counter() - t0, 2),
        u=plan.u,
        v=plan.v,
        angle_deg=plan.angle_deg,
        target_x=round(plan.block_x, 4),
        target_y=round(plan.block_y, 4),
        gripper_opening=round(grip, 1),
        sag_trim_mm=sag_mm,
        detail="성공",
    )
