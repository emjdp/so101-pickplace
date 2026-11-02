"""탑뷰 카메라로 블록을 실시간 검출한다. 빈칸: so101/detect.py

python scripts/detect_live.py blue               # calib/hsv_blue.json 사용
python scripts/detect_live.py blue --undistort   # calib/camera_top.yaml 로 왜곡 보정 후 검출

  s: 원본·결과 저장 (data/detect/)   m: 마스크 창 켜기/끄기   q: 종료
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402

import config  # noqa: E402
from so101.calib import load_intrinsics  # noqa: E402
from so101.camera import open_camera, read_frame, require_gui  # noqa: E402
from so101.detect import draw_detection, find_block, make_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("color")
ap.add_argument("--undistort", action="store_true")
ap.add_argument("--min-area", type=float, default=150.0)
args = ap.parse_args()
require_gui()

ranges = json.loads((config.ROOT / "calib" / f"hsv_{args.color}.json").read_text())["ranges"]
K = dist = None
if args.undistort:
    K, dist = load_intrinsics(config.CAMERA_INTRINSICS)

out_dir = config.DATA_DIR / "detect"
out_dir.mkdir(parents=True, exist_ok=True)
show_mask = True
cap = open_camera()
while True:
    frame = read_frame(cap)
    if K is not None:
        frame = cv2.undistort(frame, K, dist)
    mask = make_mask(frame, ranges)
    det = find_block(mask, args.min_area)
    view = draw_detection(frame.copy(), det)
    cv2.imshow("detect (s: save, m: mask, q: quit)", view)
    if show_mask:
        cv2.imshow("mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("m"):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow("mask")
    if key == ord("s"):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(out_dir / f"{stamp}_raw.jpg"), frame)
        cv2.imwrite(str(out_dir / f"{stamp}_det.jpg"), view)
        print("저장:", stamp, det if det is None else f"u={det.u:.1f} v={det.v:.1f} ang={det.angle_deg:.1f}")
cap.release()
cv2.destroyAllWindows()
