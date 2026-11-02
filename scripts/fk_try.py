"""로봇 없이 FK 를 돌려 본다. 관절각(LeRobot 기준 deg)을 넣으면 각 관절 위치와 그리퍼 끝을 출력한다.

python scripts/fk_try.py 0 0 0 0 0
python scripts/fk_try.py 0 0 0 90 0 --plot      # 막대 그림 (pip install matplotlib)
python scripts/fk_try.py 30 0 0 0 0             # shoulder_pan + 방향은 위에서 볼 때 어느 쪽?

순서: shoulder_pan shoulder_lift elbow_flex wrist_flex wrist_roll
좌표: base_link 기준 [mm]. URDF 에서 x 는 팔이 뻗는 앞쪽, z 는 위쪽이다 (확인해 볼 것).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from so101.fk import ARM_JOINTS, SO101Kinematics, lerobot_to_urdf_rad  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("q", type=float, nargs=5, help="LeRobot 관절각 [deg]")
ap.add_argument("--plot", action="store_true")
args = ap.parse_args()

kin = SO101Kinematics()
q = lerobot_to_urdf_rad(dict(zip(ARM_JOINTS, args.q)))
frames = kin.fk_frames(q)
T = frames[-1]

print(f"{'frame':22s} {'x':>8s} {'y':>8s} {'z':>8s}  [mm]")
print(f"{'base_link':22s} {0:8.1f} {0:8.1f} {0:8.1f}")
for j, F in zip(kin.joints, frames):
    x, y, z = F[:3, 3] * 1000
    print(f"{j.child:22s} {x:8.1f} {y:8.1f} {z:8.1f}")
print(f"\n그리퍼 접근 방향 (tool z 축): {np.round(T[:3, 2], 3)}")

if args.plot:
    import matplotlib.pyplot as plt

    pts = np.array([[0, 0, 0]] + [F[:3, 3] for F in frames]) * 1000
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", lw=3)
    tip, approach = pts[-1], T[:3, 2] * 40
    ax.quiver(*tip, *approach, color="r")
    for name, p in zip(["base"] + [j.child.replace("_link", "") for j in kin.joints], pts):
        ax.text(*p, name, fontsize=7)
    r = max(np.ptp(pts, axis=0).max(), 100) / 2
    c = pts.mean(axis=0)
    ax.set(xlim=(c[0] - r, c[0] + r), ylim=(c[1] - r, c[1] + r), zlim=(0, 2 * r),
           xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]", title=f"q = {args.q}")
    ax.set_box_aspect((1, 1, 1))
    plt.show()
