"""SO-101 역기구학(IK) — 닫힌 해(Closed-form 2R 평면 기구학) 및 수치 해(Damped Least-Squares).

계획서 3.2절 시스템 A의 역기구학:
1. 닫힌 해 (`ik_analytic` / `ik_topdown`):
   - shoulder_pan(q1)으로 수직 작업 평면을 목표점 (x, y)를 향해 돌린다.
   - shoulder_lift(q2), elbow_flex(q3), wrist_flex(q4) 세 관절의 회전축은 모두 수직 평면에 수직(+y)이므로
     수직 평면 위의 3R 평면 링크 기구 문제로 환원된다.
   - 그리퍼가 위에서 아래로 내려가는 각도(pitch = q2 + q3 + q4, 수직 하강 시 90°)를 지정하면
     손목 관절(wrist_flex)의 목표 위치가 정해져 2-링크(2R) 코사인 법칙으로 닫힌 해를 구할 수 있다.
   - wrist_roll(q5)은 블록 회전각에 맞춘다.

2. 수치 해 (`ik_numerical`, Levenberg-Marquardt IK):
   - 임의의 접근 방향 벡터(approach)와 관절 한계·초기값(seed)을 함께 고려해
     야코비안 기반으로 정밀하게 수렴시킨다(0.01 mm 미만의 URDF 미세 오프셋 및 비수직 접근 포함).
"""

from __future__ import annotations

import math

import numpy as np

from so101.fk import ARM_JOINTS, SO101Kinematics

# URDF base_link 기준 shoulder_pan 회전축의 (x, y) 위치 [m]
PAN_AXIS_XY = np.array([0.0388353, 0.0], dtype=float)
# 끝점 좌표계(gripper_frame_link)에서 집게가 다무는 방향 축 (0: x, 1: y, 2: z)
CLOSING_AXIS = 1


def _rot2d_xz(v: np.ndarray, angle_rad: float) -> np.ndarray:
    """수직 평면(r, z)에서 +y 축 기준 오른손 회전: r' = c*r + s*z, z' = -s*r + c*z."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]], dtype=float)


def ik_analytic(
    kin: SO101Kinematics,
    pos,
    pitch_deg: float = 90.0,
    roll_rad: float = 0.0,
    elbow_up: bool = True,
    margin_rad: float = math.radians(1.0),
) -> np.ndarray | None:
    """수직 평면 2R 기하학으로 닫힌 해(Closed-form IK)를 구한다.

    pos:       목표 그리퍼 끝 위치 [x, y, z] (base_link 기준, 단위 m)
    pitch_deg: 수평(+x)에서 아래(-z)로 숙인 그리퍼 각도 [deg]. 90° 이면 수직 하강(접근 방향 [0, 0, -1]).
    roll_rad:  wrist_roll 각도 [rad]
    elbow_up:  True 이면 팔꿈치를 위로 든 해(책상 충돌 방지), False 이면 아래로 내린 해.
    반환:      URDF 관절각 (5,) [rad] 또는 도달 불가 시 None
    """
    x, y, z = (float(v) for v in pos)
    dx, dy = x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1]

    # q1=0, q2=q3=q4=0 자세에서 각 관절의 평면(r, z) 링크 벡터 및 측방 오프셋 추출
    frames = kin.fk_frames(np.array([0.0, 0.0, 0.0, 0.0, float(roll_rad)]))
    p1 = frames[1][:3, 3]  # shoulder_lift
    p2 = frames[2][:3, 3]  # elbow_flex
    p3 = frames[3][:3, 3]  # wrist_flex
    p5 = frames[5][:3, 3]  # gripper_frame_link (TCP)

    # URDF 상 gripper_frame_joint 의 x 오프셋(-7.9 mm)이 wrist_roll 에 의해 y 방향(평면 밖) 성분 y_lat 를 만든다.
    # shoulder_pan(q1)이 회전할 때 거리 h = hypot(dx, dy) 중 y_lat 만큼이 평면 밖으로 나가므로
    # 평면 내 반지름 r = sqrt(h^2 - y_lat^2) 이고 q1 은 asin(y_lat / h) 만큼 보정된다.
    y_lat = float(p5[1])
    h = math.hypot(dx, dy)
    if h <= abs(y_lat) + 1e-6:
        return None
    r = math.sqrt(h * h - y_lat * y_lat)
    q1 = -math.atan2(dy, dx) + math.asin(y_lat / h)

    r1, z1 = float(p1[0] - PAN_AXIS_XY[0]), float(p1[2])
    v12 = np.array([p2[0] - p1[0], p2[2] - p1[2]], dtype=float)
    v23 = np.array([p3[0] - p2[0], p3[2] - p2[2]], dtype=float)
    v35 = np.array([p5[0] - p3[0], p5[2] - p3[2]], dtype=float)

    phi = math.radians(pitch_deg)  # phi = q2 + q3 + q4
    wrist_rel = np.array([r - r1, z - z1], dtype=float) - _rot2d_xz(v35, phi)

    l1 = float(np.linalg.norm(v12))
    l2 = float(np.linalg.norm(v23))
    d = float(np.linalg.norm(wrist_rel))
    if d > l1 + l2 or d < abs(l1 - l2) or d < 1e-9:
        return None

    cos_beta = (l1**2 + l2**2 - d**2) / (2.0 * l1 * l2)
    cos_alpha = (l1**2 + d**2 - l2**2) / (2.0 * l1 * d)
    beta = math.acos(float(np.clip(cos_beta, -1.0, 1.0)))
    alpha = math.acos(float(np.clip(cos_alpha, -1.0, 1.0)))

    # +z 축에서 +r 축 방향으로 잰 각도
    gamma = math.atan2(wrist_rel[0], wrist_rel[1])
    a10 = math.atan2(v12[0], v12[1])
    a20 = math.atan2(v23[0], v23[1])

    if elbow_up:
        theta1 = gamma - alpha
        theta2 = theta1 + (math.pi - beta)
    else:
        theta1 = gamma + alpha
        theta2 = theta1 - (math.pi - beta)

    q2 = theta1 - a10
    q23 = theta2 - a20
    q3 = q23 - q2
    q4 = phi - q23
    q = np.array([q1, q2, q3, q4, float(roll_rad)], dtype=float)

    lo = kin.lower + margin_rad
    hi = kin.upper - margin_rad
    if np.any(q < lo) or np.any(q > hi):
        return None
    return q


def _residual(kin: SO101Kinematics, q: np.ndarray, pos: np.ndarray, approach: np.ndarray | None, w_dir: float) -> np.ndarray:
    T = kin.fk(q)
    r = T[:3, 3] - pos
    if approach is None:
        return r
    return np.concatenate([r, w_dir * (T[:3, :3] @ kin.approach_local - approach)])


def ik_numerical(
    kin: SO101Kinematics,
    pos,
    approach=None,
    roll_rad: float | None = None,
    seed=None,
    w_dir: float = 0.08,
    tol_m: float = 0.0015,
    margin_rad: float = math.radians(2.0),
) -> np.ndarray | None:
    """수치 역기구학(Levenberg-Marquardt). 끝점 위치 pos[m]와 접근 방향 approach를 맞춘다.

    닫힌 해를 첫 출발점(seed)으로 넣고 팔꿈치를 든 여러 초기 자세에서 출발해 가장 비용이 낮은 해를 고른다.
    """
    pos = np.asarray(pos, dtype=float)
    if approach is not None:
        approach = np.asarray(approach, dtype=float)
        approach = approach / np.linalg.norm(approach)
    lo, hi = kin.lower + margin_rad, kin.upper - margin_rad

    seeds = []
    if seed is not None:
        seeds.append(np.asarray(seed, dtype=float))
    # 닫힌 해가 존재하면 최우선 초기값으로 추가
    analytic = ik_analytic(kin, pos, pitch_deg=90.0, roll_rad=roll_rad or 0.0, elbow_up=True, margin_rad=margin_rad)
    if analytic is not None:
        seeds.append(analytic)
    pan = -math.atan2(pos[1] - PAN_AXIS_XY[1], pos[0] - PAN_AXIS_XY[0])
    for lift, elbow, flex in (
        (0.0, 0.3, 1.2),
        (-0.6, 0.9, 1.2),
        (0.6, -0.4, 1.3),
        (-1.0, 1.4, 0.6),
        (0.2, 0.8, 0.3),
    ):
        seeds.append(np.array([pan, lift, elbow, flex, 0.0], dtype=float))

    best = None
    for s in seeds:
        q = np.clip(s.copy(), lo, hi)
        if roll_rad is not None:
            q[4] = float(roll_rad)
        lam = 1e-3
        err = _residual(kin, q, pos, approach, w_dir)
        for _ in range(80):
            J = np.zeros((len(err), 4))
            for k in range(4):
                dq = q.copy()
                dq[k] += 1e-5
                J[:, k] = (_residual(kin, dq, pos, approach, w_dir) - err) / 1e-5
            step = np.linalg.solve(J.T @ J + lam * np.eye(4), -J.T @ err)
            step = np.clip(step, -0.35, 0.35)
            trial = q.copy()
            trial[:4] = np.clip(q[:4] + step, lo[:4], hi[:4])
            e2 = _residual(kin, trial, pos, approach, w_dir)
            if e2 @ e2 < err @ err:
                q, err, lam = trial, e2, max(lam * 0.5, 1e-6)
                if np.linalg.norm(step) < 1e-7:
                    break
            else:
                lam *= 4.0
                if lam > 1e3:
                    break
        perr = float(np.linalg.norm(kin.tip(q) - pos))
        derr = 0.0 if approach is None else float(np.linalg.norm(kin.approach(q) - approach))
        cost = perr + 0.02 * derr
        if seed is not None:
            cost += 0.002 * float(np.linalg.norm(q[:4] - np.asarray(seed, dtype=float)[:4]))
        if perr <= tol_m and (best is None or cost < best[0]):
            best = (cost, q, perr, derr)
    return None if best is None else best[1]


def solve_grasp_ik(
    kin: SO101Kinematics,
    pos,
    roll_rad: float = 0.0,
    seed=None,
    allow_tilt: bool = True,
    method: str = "hybrid",
) -> np.ndarray | None:
    """위에서 수직으로 내려가는 자세(0°)부터 시작해, 먼 위치면 앞으로 기울여(15°~60°) 도달 가능한 해를 찾는다.

    method:
      - "analytic": 닫힌 해(2R 평면 기구학)만 사용
      - "numerical": 수치 해(LM)만 사용
      - "hybrid": 닫힌 해를 초기값으로 넣어 수치 해로 다듬음 (기본값)
    """
    pos = np.asarray(pos, dtype=float)
    radial = np.array([pos[0] - PAN_AXIS_XY[0], pos[1] - PAN_AXIS_XY[1], 0.0], dtype=float)
    n = float(np.linalg.norm(radial))
    radial = radial / n if n > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=float)

    tilts = (0.0, 15.0, 30.0, 45.0, 60.0) if allow_tilt else (0.0,)
    for tilt_deg in tilts:
        pitch_deg = 90.0 - tilt_deg
        if method == "analytic":
            q = ik_analytic(kin, pos, pitch_deg=pitch_deg, roll_rad=roll_rad, elbow_up=True)
            if q is not None:
                return q
            continue

        t = math.radians(tilt_deg)
        app = math.sin(t) * radial + np.array([0.0, 0.0, -math.cos(t)], dtype=float)
        q_seed = seed
        if method == "hybrid":
            qa = ik_analytic(kin, pos, pitch_deg=pitch_deg, roll_rad=roll_rad, elbow_up=True)
            if qa is not None:
                q_seed = qa
        q = ik_numerical(kin, pos, approach=app, roll_rad=roll_rad, seed=q_seed)
        if q is not None and float(np.linalg.norm(kin.approach(q) - app)) <= 0.25:
            return q
    return None


def roll_for_block_angle(
    kin: SO101Kinematics,
    pos,
    block_angle_world_rad: float | None,
    seed=None,
) -> float:
    """집게가 다무는 방향이 블록 회전축과 정렬되도록 wrist_roll [rad]을 고른다."""
    if block_angle_world_rad is None:
        return 0.0
    axis_xy = np.array([math.cos(block_angle_world_rad), math.sin(block_angle_world_rad)], dtype=float)
    best_roll, best_score = 0.0, -1.0
    for roll_deg in range(-90, 91, 5):
        roll = math.radians(roll_deg)
        q = solve_grasp_ik(kin, pos, roll_rad=roll, seed=seed, allow_tilt=True)
        if q is None:
            continue
        closing = kin.fk(q)[:3, CLOSING_AXIS]
        closing_xy = closing[:2] / (np.linalg.norm(closing[:2]) + 1e-9)
        score = 1.0 - abs(float(closing_xy @ axis_xy))
        if score > best_score + 1e-6:
            best_roll, best_score = roll, score
    return best_roll
