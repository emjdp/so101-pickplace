"""관절값을 실시간으로 본다. LeRobot API 를 처음 읽어 보는 코드다.

python scripts/read_joints.py              # 리더암 (손으로 자유롭게 움직임, 안전)
python scripts/read_joints.py --arm follower   # 팔로워. 토크를 끄므로 팔을 받치고 있을 것

화면의 두 숫자
  deg: LeRobot 이 캘리브레이션으로 바꾼 값. 0° = 캘리브레이션 때 기록한 범위의 가운데
  raw: 모터가 실제로 보내는 엔코더 값 (0~4095, 한 바퀴)
  gripper 는 deg 가 아니라 0(닫힘)~100(열림) 범위다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101.robot import ALL_MOTORS, connect_follower, connect_leader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["leader", "follower"], default="leader")
args = ap.parse_args()

if args.arm == "leader":
    dev = connect_leader()
else:
    dev = connect_follower()
    input("팔로워 토크를 끈다. 팔이 처지니 손으로 받치고 Enter ")
    dev.bus.disable_torque()

bus = dev.bus  # FeetechMotorsBus: 모터 레지스터를 직접 읽고 쓰는 객체
print(f"{'joint':14s} {'deg':>8s} {'raw':>6s}   range(raw)")
print("\n" * len(ALL_MOTORS), end="")
try:
    while True:
        deg = bus.sync_read("Present_Position")  # 정규화된 값
        raw = bus.sync_read("Present_Position", normalize=False)  # 엔코더 원값
        print(f"\033[{len(ALL_MOTORS)}F", end="")  # 커서를 위로 올려 같은 자리에 다시 쓴다
        for m in ALL_MOTORS:
            c = bus.calibration[m]
            unit = "%" if m == "gripper" else "°"
            print(f"{m:14s} {deg[m]:7.1f}{unit} {raw[m]:6d}   [{c.range_min}, {c.range_max}]\033[K")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    dev.disconnect()
