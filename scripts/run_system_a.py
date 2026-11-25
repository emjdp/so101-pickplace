"""시스템 A: 기구학 기반 픽앤플레이스 실행 스크립트 (계획서 3.2절).

사용법:
  # 1) 실기 1회 실행 (파란 블록 검출 -> 좌표 변환 -> IK -> 집어서 상자에 넣기)
  python scripts/run_system_a.py blue

  # 2) 실험 조건 지정 및 자동 CSV 기록 (data/experiments/trials.csv)
  python scripts/run_system_a.py blue --condition base --log

  # 3) 로봇·카메라 없이 합성 영상과 가상 로봇으로 전체 흐름 점검 (--dry-run)
  python scripts/run_system_a.py blue --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.camera import open_camera, read_frame  # noqa: E402
from so101.detect import detect_block, draw_detection  # noqa: E402
from so101.experiment import CONDITIONS, FAILURE_CAUSES, TrialRecord, append_trial  # noqa: E402
from so101.fk import SO101Kinematics  # noqa: E402
from so101.homography import HomographyCalib  # noqa: E402
from so101.pickplace import DryRunRobot, WorkspaceConfig, execute_pick_and_place  # noqa: E402
from so101.robot import connect_follower  # noqa: E402


def make_synthetic_frame(calib: HomographyCalib, target_xy=(0.24, 0.05), angle_deg: float = 25.0) -> np.ndarray:
    """--dry-run 모드용: 목표 로봇 좌표 (x, y)에 해당하는 픽셀 위치에 파란 사각형 블록을 그린다."""
    img = np.full((config.CAM_HEIGHT, config.CAM_WIDTH, 3), 120, dtype=np.uint8)
    u, v = calib.xy_to_pixel(target_xy[0], target_xy[1])
    box = cv2.boxPoints(((u, v), (36, 36), angle_deg)).astype(np.int32)
    cv2.fillPoly(img, [box], (210, 60, 20))
    return img


def load_joint_offsets() -> dict[str, float] | None:
    if config.CAMERA_EXTRINSICS.exists():
        data = json.loads(config.CAMERA_EXTRINSICS.read_text())
        return data.get("joint_offset_deg")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("color", nargs="?", default="blue", help="블록 색상 (calib/hsv_<color>.json)")
    ap.add_argument("--calib", type=Path, default=config.HOMOGRAPHY_CALIB, help="호모그래피 보정 파일 경로")
    ap.add_argument("--ik", choices=["analytic", "numerical", "hybrid"], default="hybrid", help="역기구학 방식")
    ap.add_argument("--box", type=float, nargs=3, default=list(config.BOX_XYZ_M), metavar=("X", "Y", "Z"), help="상자 위치 [m]")
    ap.add_argument("--table-z", type=float, default=config.TABLE_Z_M, help="작업대 높이 [m]")
    ap.add_argument("--condition", choices=list(CONDITIONS), default="base", help="비교 실험 조건")
    ap.add_argument("--log", action="store_true", help="실행 결과를 data/experiments/trials.csv 에 기록")
    ap.add_argument("--csv", type=Path, default=config.EXPERIMENT_CSV, help="기록할 CSV 경로")
    ap.add_argument("--dry-run", action="store_true", help="로봇·카메라 없이 가상 환경에서 전체 파이프라인 실행")
    ap.add_argument("--synth-xy", type=float, nargs=2, default=[0.24, 0.05], metavar=("X", "Y"), help="--dry-run 블록 위치 [m]")
    args = ap.parse_args()

    hsv_file = config.ROOT / "calib" / f"hsv_{args.color}.json"
    if not hsv_file.exists():
        raise SystemExit(f"HSV 설정 파일이 없습니다: {hsv_file} (python tools/hsv_tuner.py {args.color})")
    hsv_ranges = json.loads(hsv_file.read_text())["ranges"]

    if not args.calib.exists():
        raise SystemExit(f"호모그래피 파일이 없습니다: {args.calib} (python scripts/calib_homography.py)")
    calib = HomographyCalib.load(args.calib, intrinsics_path=config.CAMERA_INTRINSICS)
    kin = SO101Kinematics(offset_deg=load_joint_offsets())
    ws = WorkspaceConfig(
        table_z_m=args.table_z,
        block_height_m=config.BLOCK_HEIGHT_M,
        box_xyz_m=tuple(args.box),
        x_range_m=config.WORKSPACE_X_M,
        y_range_m=config.WORKSPACE_Y_M,
    )

    if args.dry_run:
        frame = make_synthetic_frame(calib, target_xy=tuple(args.synth_xy))
        robot = DryRunRobot()
    else:
        cap = open_camera()
        frame = read_frame(cap)
        cap.release()
        robot = connect_follower(max_relative_target=15.0)

    try:
        # HomographyCalib.pixel_to_xy()가 내부에서 K, dist로 왜곡 보정을 수행하므로
        # 원본 프레임에서 검출해야 이중 왜곡 보정을 피할 수 있다.
        proc_frame = frame

        det = detect_block(proc_frame, hsv_ranges)
        snap_dir = config.DATA_DIR / "system_a"
        snap_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(snap_dir / f"{stamp}_det.jpg"), draw_detection(proc_frame.copy(), det))

        if det is not None:
            bx, by = calib.pixel_to_xy(det.u, det.v)
            print(f"[검출 성공] 픽셀=({det.u:.1f}, {det.v:.1f}), 각도={det.angle_deg:.1f}° -> 로봇 좌표=({bx*1000:.1f}, {by*1000:.1f}) mm")
        else:
            print("[검출 실패] 화면에서 블록을 찾지 못했습니다.")

        res = execute_pick_and_place(
            robot=robot,
            kin=kin,
            calib=calib,
            det=det,
            ws=ws,
            ik_method=args.ik,
            realtime=not args.dry_run,
        )
        status = "성공" if res.ok else f"실패 ({FAILURE_CAUSES.get(res.failure_cause, res.failure_cause)})"
        extra = f" | 그리퍼={res.gripper_opening}%" if res.gripper_opening is not None else ""
        print(f"[시스템 A 결과] {status} | 소요 시간: {res.elapsed_s:.2f} s{extra} | 상세: {res.detail}")

        if args.log:
            success = 1 if res.ok else 0
            cause = res.failure_cause
            if not args.dry_run and res.ok:
                ans = input("블록이 상자 안에 들어갔습니까? [Y/n/원인코드]: ").strip().lower()
                if ans in ("n", "no", "0"):
                    success = 0
                    cause = input("실패 원인 (coord_error/grasp_fail/drop_fail) [coord_error]: ").strip() or "coord_error"
                elif ans in FAILURE_CAUSES and ans != "":
                    success = 0
                    cause = ans
            rec = append_trial(
                args.csv,
                TrialRecord(
                    timestamp="",
                    system="A",
                    condition=args.condition,
                    trial=0,
                    success=success,
                    elapsed_s=res.elapsed_s,
                    failure_cause="" if success else cause,
                    block_color=args.color,
                    u_px=round(res.u, 1) if res.u is not None else "",
                    v_px=round(res.v, 1) if res.v is not None else "",
                    target_x_m=res.target_x if res.target_x is not None else "",
                    target_y_m=res.target_y if res.target_y is not None else "",
                    note=res.detail,
                ),
            )
            print(f"[기록 저장] {args.csv} (시행 #{rec.trial}, 성공={rec.success})")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
