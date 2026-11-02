"""팔로워를 지정한 관절각으로 천천히 보낸다. 영점 확인이나 (5주차) IK 결과 확인에 쓴다.

python scripts/go_pose.py 0 0 0 0 0          # URDF 영점 자세
python scripts/go_pose.py 0 0 0 90 0         # FK 상 그리퍼가 수직으로 아래를 보는 자세
python scripts/go_pose.py 0 0 0 0 0 --speed 5

순서: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll  [LeRobot deg]
그리퍼는 지금 값을 유지한다.

동작: 현재 자세 → 목표 자세를 직선 보간 (기본 10°/s) → Enter 까지 유지 → 출발 자세로 복귀 → Enter 로 토크 해제
★ 경로 중간에 책상·카메라 기둥과 부딪히지 않는지 실행 전에 머릿속으로 그려 볼 것. 위험하면 Ctrl+C.
   (Ctrl+C 하면 토크가 꺼지면서 팔이 떨어진다. 한 손은 팔 근처에)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from so101.robot import ARM_JOINTS, connect_follower, hold_current_pose, joint_positions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("q", type=float, nargs=5)
ap.add_argument("--speed", type=float, default=10.0, help="가장 많이 움직이는 관절 기준 [deg/s]")
args = ap.parse_args()

HZ = 20
robot = connect_follower(max_relative_target=5.0)
try:
    start = hold_current_pose(robot)
    goal = dict(start, **dict(zip(ARM_JOINTS, args.q)))

    for m in ARM_JOINTS:
        c = robot.bus.calibration[m]
        half = (c.range_max - c.range_min) / 2 * 360 / 4095
        if abs(goal[m]) > half - 3:
            raise SystemExit(f"{m} 목표 {goal[m]:.1f}° 가 범위 ±{half:.1f}° 끝에 너무 가깝다")

    def move(a, b):
        span = max(abs(b[m] - a[m]) for m in ARM_JOINTS)
        n = max(1, int(span / args.speed * HZ))
        for s in np.linspace(0, 1, n + 1)[1:]:
            robot.send_action({f"{m}.pos": a[m] + s * (b[m] - a[m]) for m in a})
            time.sleep(1 / HZ)
        time.sleep(0.8)
        return joint_positions(robot.get_observation())

    print("이동:", {m: round(goal[m], 1) for m in ARM_JOINTS})
    reached = move(start, goal)
    print(f"{'joint':14s} {'명령':>7s} {'실제':>7s} {'차이':>6s}")
    for m in ARM_JOINTS:
        print(f"{m:14s} {goal[m]:7.1f} {reached[m]:7.1f} {reached[m] - goal[m]:+6.1f}")
    print("(차이가 1~2° 남는 건 중력 때문에 모터가 목표에 못 미치는 것. 어느 관절이 큰지 볼 것)")
    input("확인이 끝나면 Enter → 출발 자세로 복귀 ")
    move(reached, start)
    input("팔을 받치고 Enter → 토크 해제 ")
finally:
    robot.disconnect()
