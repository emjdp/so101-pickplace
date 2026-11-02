"""리더암으로 팔로워를 움직이면서 FK 결과를 실시간으로 보고, 자세를 기록한다. FK 실측 비교용.

python scripts/fk_live.py               # 리더 → 팔로워 텔레오퍼레이션
python scripts/fk_live.py --no-leader   # 팔로워 토크를 끄고 손으로 움직임 (받쳐 줄 사람 필요)

  Enter     : 현재 자세 기록 (0.5 s 멈춘 뒤 5번 읽어 평균)
  q + Enter : 종료 → data/fk/<시각>.csv 저장, 점 사이 거리표 출력

★ 종료 전에 리더암을 접힌(쉬는) 자세로 돌려놓을 것. 종료하면 팔로워 토크가 꺼진다.
★ 종이에 점을 찍을 때는 살짝만 닿게. 리더를 더 누르면 팔로워 모터가 책상을 밀어 무리가 간다.
"""

import argparse
import csv
import select
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.fk import SO101Kinematics  # noqa: E402
from so101.robot import ALL_MOTORS, connect_follower, connect_leader, joint_positions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--no-leader", action="store_true")
ap.add_argument("--hz", type=float, default=30.0)
args = ap.parse_args()

kin = SO101Kinematics()
follower = connect_follower(max_relative_target=None if args.no_leader else 20.0)
leader = None
if args.no_leader:
    input("팔로워 토크를 끈다. 팔을 받치고 Enter ")
    follower.bus.disable_torque()
else:
    leader = connect_leader()


def averaged_joints(n=5):
    samples = [joint_positions(follower.get_observation()) for _ in range(n)]
    return {m: float(np.mean([s[m] for s in samples])) for m in ALL_MOTORS}


records = []
print(__doc__.split("\n\n")[2])
last_print = 0.0
try:
    while True:
        t0 = time.perf_counter()
        if leader is not None:
            follower.send_action(leader.get_action())
        q = joint_positions(follower.get_observation())
        T = kin.fk_from_lerobot(q)

        if t0 - last_print > 0.2:
            x, y, z = T[:3, 3] * 1000
            a = T[:3, 2]
            sys.stdout.write(f"\r tool [mm] x={x:7.1f} y={y:7.1f} z={z:7.1f} | 접근방향 ({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}) | 기록 {len(records)}개 ")
            sys.stdout.flush()
            last_print = t0

        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower()
            if line == "q":
                break
            time.sleep(0.5)  # 멈춘 상태에서 흔들림이 가라앉게
            qa = averaged_joints()
            T = kin.fk_from_lerobot(qa)
            label = f"P{len(records) + 1}"
            records.append({"label": label, "time": time.strftime("%H:%M:%S"), **qa,
                            "x_mm": T[0, 3] * 1000, "y_mm": T[1, 3] * 1000, "z_mm": T[2, 3] * 1000,
                            "ax": T[0, 2], "ay": T[1, 2], "az": T[2, 2]})
            print(f"\n{label} 기록: x={T[0, 3] * 1000:.1f} y={T[1, 3] * 1000:.1f} z={T[2, 3] * 1000:.1f} mm  (종이에 점 찍고 {label} 라고 적기)")

        time.sleep(max(0.0, 1 / args.hz - (time.perf_counter() - t0)))
finally:
    print()
    if leader is not None:
        leader.disconnect()
    follower.disconnect()

if not records:
    sys.exit()

out = config.DATA_DIR / "fk" / f"{time.strftime('%Y%m%d_%H%M%S')}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
print("저장:", out)

P = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in records])
print("\n점 사이 거리 [mm]  — 종이 위 점 사이를 자로 재서 옆에 적는다 (문서 '4. 측정 기록' FK 표)")
print(f"{'쌍':8s} {'FK 3D':>8s} {'FK XY':>8s}")
for i, j in combinations(range(len(records)), 2):
    d3 = np.linalg.norm(P[i] - P[j])
    dxy = np.linalg.norm(P[i, :2] - P[j, :2])
    print(f"{records[i]['label']}-{records[j]['label']:5s} {d3:8.1f} {dxy:8.1f}")
print(f"\nz 평균 {P[:, 2].mean():.1f} mm, 표준편차 {P[:, 2].std():.1f} mm, 범위 {np.ptp(P[:, 2]):.1f} mm")
print("(모든 점이 같은 책상면에 닿았다면 z 는 같아야 한다. 차이가 크면 어느 관절의 영점·부호를 의심할지 생각해 볼 것)")
