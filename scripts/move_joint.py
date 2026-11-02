"""팔로워의 관절 하나를 천천히 +delta 만큼 움직였다가 되돌린다. send_action 연습용.

python scripts/move_joint.py shoulder_pan --delta 15
python scripts/move_joint.py wrist_flex --delta -20 --speed 8

관찰할 것: '+' 명령을 줬을 때 실제로 어느 쪽으로 도는가?
           → 문서 '2. 순기구학 FK' 의 부호표에 기록하고, URDF 로 예측한 방향(fk_try.py)과 비교한다.

안전장치
- delta 는 ±30° 까지만 받는다.
- 목표가 캘리브레이션 범위 끝에서 5° 안쪽이 아니면 움직이지 않는다.
- 한 번에 보내는 명령은 1 step 씩 (기본 10°/s, 20 Hz).
- 끝나면 시작 자세로 돌아온 뒤, Enter 를 눌러야 토크를 끈다 (그 전에 팔을 받칠 것).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from so101.robot import ARM_JOINTS, connect_follower, hold_current_pose, joint_positions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("joint", choices=ARM_JOINTS)
ap.add_argument("--delta", type=float, default=15.0, help="움직일 각도 [deg]")
ap.add_argument("--speed", type=float, default=10.0, help="[deg/s]")
args = ap.parse_args()
if abs(args.delta) > 30:
    raise SystemExit("delta 는 ±30° 까지")

HZ = 20
robot = connect_follower(max_relative_target=5.0)
try:
    start = hold_current_pose(robot)

    # 캘리브레이션 범위를 각도로 바꿔서 한계를 확인한다 (DEGREES 모드: 0° = 범위 가운데)
    c = robot.bus.calibration[args.joint]
    half_range_deg = (c.range_max - c.range_min) / 2 * 360 / 4095
    target = start[args.joint] + args.delta
    print(f"{args.joint}: 현재 {start[args.joint]:.1f}°, 목표 {target:.1f}°, 허용 범위 ±{half_range_deg:.1f}°")
    if abs(target) > half_range_deg - 5:
        raise SystemExit("목표가 범위 끝에 너무 가깝다. delta 부호나 크기를 바꿀 것")

    def ramp(a, b):
        n = max(1, int(abs(b - a) / args.speed * HZ))
        for x in np.linspace(a, b, n + 1)[1:]:
            action = {f"{m}.pos": start[m] for m in start}
            action[f"{args.joint}.pos"] = float(x)
            robot.send_action(action)
            time.sleep(1 / HZ)
        time.sleep(0.5)
        return joint_positions(robot.get_observation())[args.joint]

    reached = ramp(start[args.joint], target)
    print(f"도착: 명령 {target:.1f}° / 실제 {reached:.1f}° (차이 {reached - target:+.1f}°)")
    input("어느 쪽으로 움직였는지 확인하고 Enter → 되돌아감 ")
    back = ramp(target, start[args.joint])
    print(f"복귀: 명령 {start[args.joint]:.1f}° / 실제 {back:.1f}°")
    input("팔을 받치고 Enter → 토크 해제 ")
finally:
    robot.disconnect()
