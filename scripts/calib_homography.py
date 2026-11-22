"""카메라-로봇 호모그래피 보정 스크립트 (계획서 3.2절 3번).

사용법:
  # 1) 대화형 기준점 수집 및 보정 (테이블 위 표시점 4~9개에 그리퍼 끝을 대고 카메라 화면에서 클릭)
  python scripts/calib_homography.py

  # 2) 이미 저장된 카메라 외부 자세(calib/camera_extrinsics.json)에서 특정 작업 높이의 호모그래피 생성
  python scripts/calib_homography.py --from-extrinsics --plane-z 0.025

  # 3) 보정 결과 확인 (카메라 영상에 로봇 좌표 격자를 투영해 표시)
  python scripts/calib_homography.py --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.calib import load_intrinsics  # noqa: E402
from so101.camera import open_camera, read_frame, require_gui  # noqa: E402
from so101.fk import SO101Kinematics  # noqa: E402
from so101.homography import HomographyCalib, TopViewRayCalib, fit_homography  # noqa: E402
from so101.robot import ALL_MOTORS, connect_follower, connect_leader, joint_positions  # noqa: E402


def build_from_extrinsics(plane_z_m: float, out_path: Path) -> None:
    ray_calib = TopViewRayCalib.from_files(config.CAMERA_INTRINSICS, config.CAMERA_EXTRINSICS)
    h_calib = ray_calib.to_homography(plane_z_m=plane_z_m)
    h_calib.save(out_path, intrinsics_path=config.CAMERA_INTRINSICS)
    print(f"외부 파라미터 기반 호모그래피 저장 완료: {out_path}")
    print(f"  평면 높이 z = {plane_z_m * 1000:.1f} mm, 변환 잔차 RMS = {h_calib.rms_mm:.3f} mm")


def verify_live(calib_path: Path) -> None:
    require_gui()
    h_calib = HomographyCalib.load(calib_path, intrinsics_path=config.CAMERA_INTRINSICS)
    cap = open_camera()
    clicked = None

    def on_click(event, x, y, *_):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            rx, ry = h_calib.pixel_to_xy(x, y)
            clicked = (x, y, rx, ry)
            print(f"픽셀 (u={x}, v={y}) -> 로봇 베이스 (x={rx * 1000:.1f} mm, y={ry * 1000:.1f} mm)")

    win = "homography verify (click to test, q: quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    while True:
        frame = read_frame(cap)
        if h_calib.K is not None and h_calib.dist is not None:
            view = cv2.undistort(frame, h_calib.K, h_calib.dist)
        else:
            view = frame.copy()

        # 로봇 작업 영역 격자 표시 (x: 0.15~0.35 m, y: -0.15~0.15 m)
        for x in np.linspace(0.15, 0.35, 5):
            for y in np.linspace(-0.15, 0.15, 5):
                u, v = h_calib.xy_to_pixel(float(x), float(y))
                if 0 <= u < view.shape[1] and 0 <= v < view.shape[0]:
                    cv2.circle(view, (int(round(u)), int(round(v))), 3, (255, 200, 0), -1)
        if clicked:
            cu, cv_y, rx, ry = clicked
            cv2.drawMarker(view, (cu, cv_y), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                view,
                f"({rx * 1000:.0f}, {ry * 1000:.0f}) mm",
                (cu + 8, cv_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        cv2.imshow(win, view)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def interactive_collect(out_path: Path, use_undistort: bool, no_leader: bool) -> None:
    require_gui()
    K = dist = None
    if use_undistort and config.CAMERA_INTRINSICS.exists():
        K, dist = load_intrinsics(config.CAMERA_INTRINSICS)

    kin = SO101Kinematics()
    follower = connect_follower(max_relative_target=None if no_leader else 20.0)
    leader = None
    if no_leader:
        input("팔로워 토크를 끕니다. 손으로 팔을 받치고 Enter를 누르세요: ")
        follower.bus.disable_torque()
    else:
        leader = connect_leader()

    cap = open_camera()
    points: list[dict] = []
    latest_click: tuple[int, int] | None = None

    def on_click(event, x, y, *_):
        nonlocal latest_click
        if event == cv2.EVENT_LBUTTONDOWN:
            latest_click = (int(x), int(y))

    win = "homography calib (1. move tip -> 2. click tip -> Space: add, s: solve & save, q: quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    print("1) 그리퍼 끝을 테이블 기준점에 맞추고, 2) 카메라 화면에서 그 점을 마우스로 클릭한 뒤 Space 키를 누르세요 (4~9점).")
    print("점들을 모은 뒤 's' 키를 누르면 호모그래피 H를 계산해 저장합니다.")

    try:
        while True:
            if leader is not None:
                follower.send_action(leader.get_action())
            q = joint_positions(follower.get_observation())
            tip = kin.fk_from_lerobot(q)[:3, 3]

            frame = read_frame(cap)
            view = frame.copy()
            for idx, pt in enumerate(points, 1):
                cv2.circle(view, (int(pt["u"]), int(pt["v"])), 4, (0, 255, 255), -1)
                cv2.putText(
                    view,
                    f"P{idx}",
                    (int(pt["u"]) + 6, int(pt["v"]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
            if latest_click is not None:
                cv2.drawMarker(view, latest_click, (0, 255, 0), cv2.MARKER_CROSS, 14, 2)

            status = f"FK [mm] x={tip[0]*1000:.1f} y={tip[1]*1000:.1f} z={tip[2]*1000:.1f} | saved={len(points)}"
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(win, view)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and latest_click is not None:
                samples = [joint_positions(follower.get_observation()) for _ in range(5)]
                qa = {m: float(np.mean([s[m] for s in samples])) for m in ALL_MOTORS}
                t_pos = kin.fk_from_lerobot(qa)[:3, 3]
                u, v = latest_click
                points.append(
                    {
                        "u": float(u),
                        "v": float(v),
                        "x": round(float(t_pos[0]), 5),
                        "y": round(float(t_pos[1]), 5),
                        "z": round(float(t_pos[2]), 5),
                    }
                )
                print(
                    f"추가 P{len(points)}: pixel=({u}, {v}) <-> robot=({t_pos[0]*1000:.1f}, {t_pos[1]*1000:.1f}, z={t_pos[2]*1000:.1f}) mm"
                )
                latest_click = None
            if key == ord("s"):
                if len(points) < 4:
                    print(f"최소 4점이 필요합니다 (현재 {len(points)}점)")
                    continue
                uv = np.array([[p["u"], p["v"]] for p in points], dtype=np.float64)
                xy = np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)
                plane_z = float(np.mean([p["z"] for p in points]))
                H, errs = fit_homography(uv, xy, K=K, dist=dist)
                rms_mm = float(np.sqrt(np.mean(errs**2)) * 1000)
                max_mm = float(np.max(errs) * 1000)
                calib = HomographyCalib(H=H, plane_z_m=plane_z, K=K, dist=dist, rms_mm=rms_mm, max_mm=max_mm)
                calib.save(out_path, points=points, intrinsics_path=config.CAMERA_INTRINSICS if K is not None else None)
                print(f"\n저장 완료: {out_path}")
                print(f"  기준점 {len(points)}개, 평균 높이 z={plane_z*1000:.1f} mm, 오차 RMS={rms_mm:.2f} mm, 최대={max_mm:.2f} mm")
                for i, e in enumerate(errs, 1):
                    print(f"  P{i}: 오차 {e * 1000:.2f} mm")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if leader is not None:
            leader.disconnect()
        follower.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=config.HOMOGRAPHY_CALIB)
    ap.add_argument("--from-extrinsics", action="store_true", help="calib/camera_extrinsics.json 에서 호모그래피 생성")
    ap.add_argument("--plane-z", type=float, default=config.TABLE_Z_M + config.BLOCK_HEIGHT_M, help="변환 평면 높이 [m]")
    ap.add_argument("--verify", action="store_true", help="저장된 호모그래피를 실시간 카메라 영상으로 확인")
    ap.add_argument("--no-undistort", action="store_true", help="내부 보정 왜곡 펴기를 쓰지 않음")
    ap.add_argument("--no-leader", action="store_true", help="리더암 없이 팔로워를 손으로 움직여 측정")
    args = ap.parse_args()

    if args.from_extrinsics:
        build_from_extrinsics(args.plane_z, args.out)
    elif args.verify:
        verify_live(args.out)
    else:
        interactive_collect(args.out, use_undistort=not args.no_undistort, no_leader=args.no_leader)


if __name__ == "__main__":
    main()
