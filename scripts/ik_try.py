"""역기구학(IK) 계산 및 FK 재투영 검증 스크립트.

로봇 없이 목표 위치 (x, y, z) [mm] 와 손목 회전각을 넣어 닫힌 해(analytic)와 수치 해(numerical)를
비교하고, 필요하면 3D 막대 그림(--plot)이나 실제 로봇 이동(--go)으로 확인한다.

사용법:
  python scripts/ik_try.py 220 0 25
  python scripts/ik_try.py 250 80 30 --roll 15 --plot
  python scripts/ik_try.py 220 -60 25 --go
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from so101.fk import ARM_JOINTS, SO101Kinematics  # noqa: E402
from so101.ik import ik_analytic, ik_numerical, solve_grasp_ik  # noqa: E402
from so101.pickplace import move_smooth  # noqa: E402
from so101.robot import connect_follower, hold_current_pose  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("xyz_mm", type=float, nargs=3, help="목표 위치 x y z [mm] (base_link 기준)")
ap.add_argument("--pitch", type=float, default=90.0, help="그리퍼 하강 각도 [deg] (90=수직 하강)")
ap.add_argument("--roll", type=float, default=0.0, help="wrist_roll 각도 [deg]")
ap.add_argument("--plot", action="store_true", help="3D 자세 막대 그림 표시")
ap.add_argument("--go", action="store_true", help="실제 팔로워 로봇을 해당 해로 이동")
args = ap.parse_args()

kin = SO101Kinematics()
target_m = np.array(args.xyz_mm, dtype=float) / 1000.0
roll_rad = math.radians(args.roll)

print(f"목표 위치: x={args.xyz_mm[0]:.1f} mm, y={args.xyz_mm[1]:.1f} mm, z={args.xyz_mm[2]:.1f} mm (pitch={args.pitch:.1f}°, roll={args.roll:.1f}°)")

q_ana = ik_analytic(kin, target_m, pitch_deg=args.pitch, roll_rad=roll_rad, elbow_up=True)
if q_ana is not None:
    deg_ana = kin.to_lerobot(q_ana)
    tip_ana = kin.tip(q_ana) * 1000
    err_ana = float(np.linalg.norm(tip_ana - args.xyz_mm))
    print("\n[1] 닫힌 해 (2R 평면 기구학, elbow-up):")
    print("  LeRobot 관절각 [deg]:", {k: round(v, 2) for k, v in deg_ana.items()})
    print(f"  FK 검산 끝점 [mm]  : x={tip_ana[0]:.2f}, y={tip_ana[1]:.2f}, z={tip_ana[2]:.2f} (오차 {err_ana:.3f} mm)")
else:
    print("\n[1] 닫힌 해: 해당 각도에서 도달 불가")

q_hyb = solve_grasp_ik(kin, target_m, roll_rad=roll_rad, allow_tilt=True, method="hybrid")
if q_hyb is not None:
    deg_hyb = kin.to_lerobot(q_hyb)
    tip_hyb = kin.tip(q_hyb) * 1000
    app_hyb = kin.approach(q_hyb)
    err_hyb = float(np.linalg.norm(tip_hyb - args.xyz_mm))
    print("\n[2] 하이브리드 해 (닫힌 해 초기화 + LM 수치 최적화, 자동 기울임 포함):")
    print("  LeRobot 관절각 [deg]:", {k: round(v, 2) for k, v in deg_hyb.items()})
    print(f"  FK 검산 끝점 [mm]  : x={tip_hyb[0]:.2f}, y={tip_hyb[1]:.2f}, z={tip_hyb[2]:.2f} (오차 {err_hyb:.3f} mm)")
    print(f"  접근 방향 벡터     : ({app_hyb[0]:+.3f}, {app_hyb[1]:+.3f}, {app_hyb[2]:+.3f})")
else:
    print("\n[2] 하이브리드 해: 도달 불가 (작업 범위 밖)")
    sys.exit(1)

q_use = q_ana if q_ana is not None else q_hyb
deg_use = kin.to_lerobot(q_use)

if args.plot:
    import matplotlib.pyplot as plt

    frames = kin.fk_frames(q_use)
    pts = np.array([[0, 0, 0]] + [F[:3, 3] for F in frames]) * 1000
    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", lw=3, label="arm")
    ax.scatter(*args.xyz_mm, color="g", s=60, label="target")
    tip, approach = pts[-1], frames[-1][:3, 2] * 40
    ax.quiver(*tip, *approach, color="r")
    r = max(np.ptp(pts, axis=0).max(), 120) / 2
    c = pts.mean(axis=0)
    ax.set(
        xlim=(c[0] - r, c[0] + r),
        ylim=(c[1] - r, c[1] + r),
        zlim=(0, 2 * r),
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        title=f"IK target = {args.xyz_mm} mm",
    )
    ax.legend()
    ax.set_box_aspect((1, 1, 1))
    plt.show()

if args.go:
    robot = connect_follower(max_relative_target=8.0)
    try:
        start = hold_current_pose(robot)
        print("\n목표 자세로 이동합니다...")
        move_smooth(robot, deg_use, speed_deg_s=25.0)
        input("확인 후 Enter를 누르면 출발 자세로 복귀합니다: ")
        move_smooth(robot, start, speed_deg_s=25.0)
        input("팔을 손으로 받치고 Enter를 누르면 토크를 해제합니다: ")
    finally:
        robot.disconnect()
